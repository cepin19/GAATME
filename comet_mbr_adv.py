# -*- coding: utf-8 -*-
# Copyright (C) 2020 Unbabel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Command for Minimum Bayes Risk Decoding.
========================================

This script is inspired in Chantal Amrhein script used in:
    Title: Identifying Weaknesses in Machine Translation Metrics Through Minimum Bayes Risk Decoding: A Case Study for COMET
    URL: https://arxiv.org/abs/2202.05148

optional arguments:
  -h, --help            Show this help message and exit.
  -s SOURCES, --sources SOURCES
                        (type: Path_fr, default: null)
  -t TRANSLATIONS, --translations TRANSLATIONS
                        (type: Path_fr, default: null)
  --batch_size BATCH_SIZE
                        (type: int, default: 8)
  --num_samples NUM_SAMPLES
                        (required, type: int)
  --model MODEL         COMET model to be used. (type: str, default: wmt20-comet-da)
  --model_storage_path MODEL_STORAGE_PATH
                        Path to the directory where models will be stored. By default its saved in ~/.cache/torch/unbabel_comet/ (default: null)
  -o OUTPUT, --output OUTPUT
                        Best candidates after running MBR decoding. (type: str, default: mbr_result.txt)
"""
import itertools
import os, sys
from typing import List, Tuple
import json
import numpy as np
import torch
from jsonargparse import ArgumentParser
from jsonargparse.typing import Path_fr
from tqdm import tqdm

import sacrebleu
from numpy.random import randint
from numpy.random import rand
import random
from sacrebleu.metrics import CHRF,BLEU
from timeit import default_timer as timer
import logging
# from nltk.tokenize.treebank import TreebankWordDetokenizer, TreebankWordTokenizer
from comet.models import RegressionMetric, download_model, load_from_checkpoint
import sacremoses

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.models.nlp.unite.configuration import InputFormat

import tensorflow as tf
from  bleurt import score as bleurt_score
logging.basicConfig()
logging.basicConfig(level=logging.DEBUG)
logging.root.setLevel(logging.DEBUG)
# Create a handler
c_handler = logging.StreamHandler()
logger = logging.getLogger("GA")
logger.addHandler(c_handler)

logger.setLevel(logging.DEBUG)

tok = sacremoses.MosesTokenizer(lang='en')
detok = sacremoses.MosesDetokenizer(lang='en')
max_scores = {"fitness_chrf": 99.9, "fitness_bleu": 99.9, "fitness_bleu_multiref": 99.9, "fitness_chrf_multiref": 99.9}
c = {}

class Solution:
    def __init__(self, chromosome, original=True):
        self.chromosome = chromosome
        self.chromosome_detok = detok.detokenize(chromosome).strip().replace('□',' ')
        if self.chromosome_detok == '':
            self.chromosome_detok = 'EMPTY'  # workaround for some metrics that don't accept empty lines
        self.original = original
        self.score = None
class GA:
    def __init__(self, objective, src, init_pop, max_len, refs, possible_tgts, cfg, model = None,
    model_qe = None, pseudo_refs = None, model_unite=None, model_bleurt=None
    ):

        self.objective = getattr(self,objective)

        self.src = src
        self.possible_tgts = possible_tgts
        self.init_pop = init_pop
        self.refs = refs
        self.max_len = max_len
        self.model= model
        self.model_qe= model_qe
        self.pseudo_refs= pseudo_refs
        self.unite_model=model_unite

        self.model_bleurt=model_bleurt
        self.generations = cfg.generations
        self.mutation_rate = cfg.mutation
        self.crossover_rate = cfg.crossover
        self.n_pop = len(self.init_pop)

        self.caching= cfg.cache_embeddings
        self.caching_scores= cfg.cache_scores
        self.remove_identical_pseudorefs= cfg.remove_identical_pseudorefs
        self.log= {}
        self.deletions= not cfg.no_empty
        self.bleu_w= cfg.bleu_w
        self.chrf_w= cfg.chrf_w
        self.comet_mbr_w=cfg.comet_mbr_w
        self.comet_qe_w= cfg.comet_qe_w
        self.unite_w= cfg.unite_w
        self.bleurt_w= cfg.bleurt_w
        self.comet_mbr_batch_size= cfg.comet_mbr_batch_size
        self.qe_batch_size= cfg.qe_batch_size

        self.emb_cache = {}
        self.emb_cache_qe = {}
        self.score_cache = {}
        if model is not None:
            self.src_embeddings = self.build_single_embeddings([self.src], self.model, 1, self.emb_cache,
                                                        caching=self.caching)
            self.pseudo_ref_embeddings = self.build_single_embeddings(self.pseudo_refs, self.model, len(self.pseudo_refs),
                                                               self.emb_cache, caching=self.caching)
        else:
            self.src_embeddings=self.pseudo_ref_embeddings=None
    def build_single_embeddings(self,
                                sources: List[str],
                                model: RegressionMetric,
                                batch_size: int,
                                cache,
                                caching=True
                                ) -> Tuple[torch.Tensor]:
        """Tokenization and respective encoding of source and translation sentences using
        a RegressionMetric model.

        :param sources: List of source sentences.
        :param translations: List of translation sentences.
        :param model: RegressionMetric model that will be used to embed sentences.
        :param batch_size: batch size used during encoding.

        :return: source and MT embeddings.
        """
        # TODO: Optimize this function to have faster MBR decoding!
        cache_idx = []
        cache_emb = []
        new_sources = []
        # logger.warning(len(sources))
        # logger.warning("cache size: {}".format(len(cache)))
        # logger.warning(cache)
        batch_size = min(batch_size, 250)  # to fit on A100
        for i, s in enumerate(sources):
            if s in cache and caching:
                cache_idx.append(i)
                cache_emb.append(cache[s])
            else:
                new_sources.append(s)
        # logger.warning("cache idx: {}".format(cache_idx))
        logger.warning("New (uncached) embs:")
        logger.warning(len(new_sources))
        logger.warning(len(self.emb_cache))

        if len(new_sources) != 0:  # TODO: WHY THIS HAPPENS!!!!!!

            src_batches = [
                new_sources[i: i + batch_size] for i in range(0, len(new_sources), batch_size)
            ]
            # logger.warning(batch_size)
            with torch.no_grad():
                src_inputs = [model.encoder.prepare_sample(batch) for batch in src_batches]
                src_embeddings = []
                for batch in src_inputs:
                    input_ids = batch["input_ids"].to(model.device)
                    # logger.warning(input_ids)
                    attention_mask = batch["attention_mask"].to(model.device)
                    src_embeddings.append(
                        model.get_sentence_embedding(input_ids, attention_mask)
                    )

            src_embeddings = torch.vstack(src_embeddings)
        #  logger.warning(src_embeddings)
        #   logger.warning(src_embeddings.shape)
        final_embs = []
        cache_i = 0
        computed_i = 0
        for i in range(len(sources)):
            if i in cache_idx and caching:
                final_embs.append(cache_emb[cache_i])
                cache_i += 1
            else:
                #      logger.warning("appending {}".format(src_embeddings[computed_i]))
                final_embs.append(src_embeddings[computed_i])
                cache[sources[i]] = src_embeddings[computed_i]
                computed_i += 1
        final_embs = torch.vstack(final_embs)
        # logger.warning("final embs:{}".format(final_embs))
        return final_embs

    def mbr_decoding(self,
                     src_embeddings: torch.Tensor, mt_embeddings: torch.Tensor, pseudo_refs: torch.Tensor,
                     model: RegressionMetric) -> torch.Tensor:
        """Performs MBR Decoding for each translation for a given source.

        :param src_embeddings: Embeddings of source sentences.
        :param mt_embeddings: Embeddings of MT sentences.
        :param model: RegressionMetric Model.

        :return:
            Returns a [n_sent x num_samples] matrix M where each line represents a source sentence
            and each column a given sample.
            M[i][j] is the MBR score of sample j for source i.
        """
        n_sent, num_samples, _ = mt_embeddings.shape
        ref_samples = pseudo_refs.shape[0]
        mbr_matrix = torch.zeros(n_sent, mt_embeddings.shape[1])

        with torch.no_grad():
            # Loop over all source sentences
            for i in range(mbr_matrix.shape[0]):
                source = src_embeddings[i, :].repeat(ref_samples, 1)
                # Loop over all hypothesis
                for j in tqdm(range(mt_embeddings.shape[1]), desc="MBR Scores...", dynamic_ncols=True
                              ):
                    # print(j)
                    translation = mt_embeddings[i, j, :].repeat(ref_samples, 1)

                    # Score current hypothesis against pseudo refs
                    # print("shapes: {}{}{}".format(source.shape, translation.shape, pseudo_refs.shape))

                    scores = model.estimate(source, translation, pseudo_refs)[
                        "score"
                    ]#.squeeze(1)
                    # scores = torch.cat([scores[0:j], scores[j + 1 :]]) #this is to ignore self-referenced scoring, but we dont care now (and it doesn't make sense when translations and pseudo refs are not (necesarrily) the same
                    # TODO: solve!
                    mbr_matrix[i, j] = scores.mean()
        return mbr_matrix

    def fitness_bleu_multiref(self, src, src_embeddings, solutions, pseudo_refs=None, remove_identical_pseudorefs=True, **kwargs):
        scores = []
        bleu = BLEU()
        ref_cache = bleu._cache_references([pseudo_refs])
        bleu._ref_cache = ref_cache
        #logger.info("ref: {}".format(pseudo_refs))
        for solution in solutions:
            so = solution.chromosome_detok
            # Filter senences which were the same in translation and pseudo refs
            #    pseudo_refs_f = [r.strip() for r in pseudo_refs if r != "#EMPTY"]
            refs = []
            for pr in pseudo_refs:
                if len(pseudo_refs) > 1 and remove_identical_pseudorefs and so == pr.strip() and solution.original is True:
                    # logger.error("skipping ref {} for hyp {}".format(pr,so))
                    bleu._ref_cache=ref_cache.copy().remove(i)

                    continue
                else:
                    refs.append(pr)
            #s = sacrebleu.sentence_bleu(so, refs, smooth_method='exp')
            stats=bleu._extract_corpus_statistics([so], None)
            s=bleu._aggregate_and_compute(stats)
            scores.append(s.score)
        return scores

    def fitness_chrf_multiref(self, src, src_embeddings, solutions, pseudo_refs=None, remove_identical_pseudorefs=True, **kwargs):
        scores = []
        chrf=CHRF()
        ref_cache=chrf._cache_references([pseudo_refs])
        chrf._ref_cache=ref_cache
        for solution in solutions:
            chrf._ref_cache = ref_cache

            so = solution.chromosome_detok
            # Filter senences which were the same in translation and pseudo refs
            #    pseudo_refs_f = [r.strip() for r in pseudo_refs if r != "#EMPTY"]
            refs = []
            for i,pr in enumerate(pseudo_refs):
                if len(pseudo_refs) > 1 and remove_identical_pseudorefs and so == pr.strip() and solution.original is True:
                    # logger.error("skipping ref {} for hyp {}".format(pr,so))
                    chrf._ref_cache=ref_cache.copy().remove(i)

                    continue

                else:
                    refs.append(pr)

            #s = sacrebleu.sentence_chrf(so, refs)
            stats=chrf._extract_corpus_statistics([so], None)
            s=chrf._aggregate_and_compute(stats)
            scores.append(s.score)
        return scores
    def fitness_bleu(self, src, src_embeddings, solutions, pseudo_refs=None, remove_identical_pseudorefs=True, **kwargs):
        # print(' '.join(solution))
        scores = []
        bleu = BLEU()
        ref_cache = bleu._cache_references([pseudo_refs])
        for solution in solutions:
            if solution.score is not None:
                scores.append(solution.score)
            else:
                so = solution.chromosome_detok
                sol_scores = []
                # Filter senences which were the same in translation and pseudo refs
                #    pseudo_refs_f = [r.strip() for r in pseudo_refs if r != "#EMPTY"]
                for i,pr in enumerate(pseudo_refs):
                    if len(pseudo_refs) > 1 and remove_identical_pseudorefs and so == pr.strip() and solution.original is True:
                        # logger.error("skipping ref {} for hyp {}".format(pr,so))
                        continue
                    bleu._ref_cache = [ref_cache[i]]
                    stats = bleu._extract_corpus_statistics([so], None)
                    s = bleu._aggregate_and_compute(stats)
                    #s = sacrebleu.sentence_bleu(so, [pr], smooth_method='exp')
                    sol_scores.append(s.score)
                avg = sum(sol_scores) / len(sol_scores)
                scores.append(avg)
                solution.score = avg
        return scores

    def fitness_chrf(self, src, src_embeddings, solutions, pseudo_refs=None, remove_identical_pseudorefs=True, **kwargs):
        # print(' '.join(solution))
        scores = []
        chrf=CHRF()
        ref_cache=chrf._cache_references([pseudo_refs])
        for solution in solutions:
            if solution.score is not None:
                scores.append(solution.score)
            else:
                so = solution.chromosome_detok
                sol_scores = []
                # Filter senences which were the same in translation and pseudo refs
                #    pseudo_refs_f = [r.strip() for r in pseudo_refs if r != "#EMPTY"]
                for pr in pseudo_refs:
                    if len(pseudo_refs) > 1 and remove_identical_pseudorefs and so == pr.strip() and solution.original is True:
                        # logger.error("skipping ref {} for hyp {}".format(pr,so))
                        continue
                    chrf._ref_cache = [ref_cache[i]]
                    stats = chrf._extract_corpus_statistics([so], None)
                    s = chrf._aggregate_and_compute(stats)
                    #s = sacrebleu.sentence_chrf(so, [pr])
                    sol_scores.append(s.score)
                avg = sum(sol_scores) / len(sol_scores)
                scores.append(avg)
                solution.score = avg
        return scores

    def fitness_comet_ref(self, src, src_embeddings, solutions, ref, model):
        solutions = [solution.chromosome_detok for solution in solutions if solution.chromosome_detok != '']

        data = [
            {
                "src": src,
                "mt": solution.strip(),  # ,
                "ref": ref[0]
            } for solution in solutions]
        logger.warning("ref {}".format(ref[0]))
        #logger.warning("data: {}".format(data))
        scores= model.predict(data, 32, gpus=1, progress_bar=False)['scores']
        return scores

    def fitness_comet_qe(self, src, solutions, model_qe, **kwargs):
        #solutions = [list(filter(lambda t: t != '', solution)) for solution in solutions]
        solutions = [solution.chromosome_detok for solution in solutions if solution.chromosome_detok != '']
        data = [
            {
                "src": src,
                "mt": solution.strip()  # ,
                #            "ref": ref
            } for solution in solutions]
        seg_scores, sys_score = model_qe.predict(data, 64, gpus=1, progress_bar=False)
        return seg_scores
    def fitness_comet_qe_new(self, src, solutions, model_qe, batch_size=32, **kwargs):
        solutions = [solution.chromosome_detok for solution in solutions if solution.chromosome_detok != '']
        data = {
                "src": [[src.strip()] for _ in solutions],
                "mt": [[solution.strip()] for solution in solutions]  # ,
                #            "ref": ref
            }
        

            # Flatten all data to score across multiple GPUs
        for k, v in data.items():
            data[k] = list(itertools.chain(*v))

        data = [dict(zip(data, t)) for t in zip(*data.values())]
        outputs = model_qe.predict(
                samples=data,
                batch_size=batch_size,
                gpus=1,
                accelerator="auto",
                length_batching=True,
            )
        seg_scores = outputs.scores
        return seg_scores
    
    
    def fitness_comet_mbr(self, src, src_embeddings, solutions, model=None, model_qe=None, pseudo_refs=None,
                          pseudo_ref_embeddings=None, caching=True, caching_scores=True, remove_identical_pseudorefs=True):
        solutions = [solution.chromosome_detok for solution in solutions if solution.chromosome_detok != '']

        start = timer()

        mt_embeddings = self.build_single_embeddings(solutions, model, len(solutions), self.emb_cache, caching=caching)
        end = timer()
        logger.info("Computing embeddings for the solutions took {} s".format(end - start))
        mt_embeddings = mt_embeddings.reshape(1, len(solutions), -1)
        # return [1.0]*len(solutions)
        return self.mbr_decoding(src_embeddings, mt_embeddings, pseudo_ref_embeddings, model)[
            0].tolist()  # only one source, so [0]

    def fitness_unite_ref(self, src, src_embeddings, solutions, ref, model_unite):
        solutions = [solution.chromosome_detok for solution in solutions if solution.chromosome_detok != '']
        batch_size=64
        total_scores=[]
        for i in range(0, len(solutions), batch_size):
            input ={
                "src": [src]*len(solutions[i:i+batch_size]),
                "hyp": solutions[i:i+batch_size],  # ,
                "ref": [ref[0]]*len(solutions[i:i+batch_size])
            }

            total_scores.extend(model_unite(input)['score'])
        return (total_scores)

    def fitness_bleurt_ref(self, src, src_embeddings, solutions, ref, model_bleurt):
        solutions = [solution.chromosome_detok for solution in solutions if solution.chromosome_detok != '']
        scores = model_bleurt.score(references=[ref[0]]*len(solutions), candidates=solutions, batch_size=64)

        return(scores)

    def fitness_comet_mbr_and_qe_and_bleu_and_chrf_w(self, src, src_embeddings, solutions, model=None, model_qe=None, pseudo_refs=None,
                                 pseudo_ref_embeddings=None, caching=True, caching_scores=True, remove_identical_pseudorefs=True, bleu_w=1, chrf_w=1, comet_mbr_w=1, comet_qe_w=1):
        #bleus=self.fitness_bleu_multiref(src, src_embeddings, solutions, pseudo_refs, remove_identical_pseudorefs)
        bleus=self.fitness_bleu(src, src_embeddings, solutions, pseudo_refs, remove_identical_pseudorefs)
        chrfs=self.fitness_chrf(src, src_embeddings, solutions, pseudo_refs, remove_identical_pseudorefs)
        mbr_and_qe=self.fitness_comet_mbr_and_qe(src, src_embeddings, solutions, model, model_qe, pseudo_refs,
                                 pseudo_ref_embeddings, caching, caching_scores, remove_identical_pseudorefs,comet_mbr_w=comet_mbr_w, comet_qe_w=comet_qe_w)
        logger.warning("weights: {} {} {} {}".format(bleu_w, chrf_w, comet_qe_w, comet_mbr_w))
        return ((np.asarray(bleus)*bleu_w)/100)+np.asarray(mbr_and_qe)+((np.asarray(chrfs)*chrf_w)/100)

    def fitness_ref_combined_multiref(self, src, src_embeddings, solutions, model=None, model_qe=None, pseudo_refs=None,
                                 pseudo_ref_embeddings=None, caching=True, caching_scores=True,
                                remove_identical_pseudorefs=True):

        cache_idx = []
        cached_scores = []
        new_solutions = []
        if caching_scores:
            for i, s in enumerate(solutions):
                #if s in self.score_cache and caching:
                if s.score is not None and caching:
                    cache_idx.append(i)
                    #cached_scores.append(self.score_cache[s])
                    cached_scores.append(s.score)
                else:
                    new_solutions.append(s)
        else:
            new_solutions = solutions

        logger.info(f"\n{len(new_solutions)}/{len(solutions)} of solutions are new.")

        if new_solutions:
            if self.bleu_w!=0:
                bleus=self.fitness_bleu_multiref(src, src_embeddings, new_solutions, pseudo_refs, remove_identical_pseudorefs)
            else:
                bleus=[0.0]*len(new_solutions)
            if self.chrf_w!=0:
                chrfs=self.fitness_chrf_multiref(src, src_embeddings, new_solutions, pseudo_refs, remove_identical_pseudorefs)
            else:
                chrfs=[0.0]*len(new_solutions)

            if self.comet_mbr_w!=0:
                start = timer()
                comet=self.fitness_comet_ref(src, src_embeddings, new_solutions, pseudo_refs, model)
                end = timer()
                logger.info("Computing COMET  scores took {} s".format(end - start))
            else:
                comet=[0.0]*len(new_solutions)
            if self.comet_qe_w!=0:
                start = timer()

                comet_qe=self.fitness_comet_qe_new(src, new_solutions, model_qe)
                end = timer()
                logger.info("Computing COMET-QE  scores took {} s".format(end - start))
            else:
                comet_qe=[0.0]*len(new_solutions)
            if self.unite_w!=0:
                start = timer()

                unite=self.fitness_unite_ref(src, src_embeddings, new_solutions, pseudo_refs, self.unite_model)
                end = timer()
                logger.info("Computing UniTE  scores took {} s".format(end - start))
            else:
                unite=[0.0]*len(new_solutions)
            if self.bleurt_w!=0:
                start = timer()
                bleurt=self.fitness_bleurt_ref(src, src_embeddings, new_solutions, pseudo_refs, self.model_bleurt)
                end = timer()
                logger.info("Computing BLEURT scores took {} s".format(end - start))
            else:
                bleurt=[0.0]*len(new_solutions)

            total_scores = (((np.asarray(bleus) * self.bleu_w) / 100) + np.asarray(comet_qe)*self.comet_qe_w + (
                        (np.asarray(chrfs) * self.chrf_w) / 100) + np.asarray(comet)*self.comet_mbr_w +
                            np.asarray(unite)*self.unite_w + np.asarray(bleurt)*self.bleurt_w)
        else:
            total_scores = []
        computed_i = 0
        cache_i = 0
        final_scores = []
        if caching_scores:
            for i in range(len(solutions)):
                if i in cache_idx:
                    solutions[i].score = cached_scores[cache_i]
                    final_scores.append(cached_scores[cache_i])
                    cache_i += 1
                else:
                    solutions[i].score = total_scores[computed_i]
                    final_scores.append(total_scores[computed_i])
                    # self.score_cache[solutionsd[i]] = total_scores[computed_i]
                    computed_i += 1
        else:
            final_scores = total_scores
        return final_scores


    def fitness_ref_combined_singleref(self, src, src_embeddings, solutions, model=None, model_qe=None, pseudo_refs=None,
                                 pseudo_ref_embeddings=None, caching=True, caching_scores=True,
                                remove_identical_pseudorefs=False):

        cache_idx = []
        cached_scores = []
        new_solutions = []
        if caching_scores:
            for i, s in enumerate(solutions):
                #if s in self.score_cache and caching:
                if s.score is not None and caching:
                    cache_idx.append(i)
                    #cached_scores.append(self.score_cache[s])
                    cached_scores.append(s.score)
                else:
                    new_solutions.append(s)
        else:
            new_solutions = solutions

        logger.info(f"\n{len(new_solutions)}/{len(solutions)} of solutions are new.")

        if new_solutions:
            if self.bleu_w!=0:
                bleus=self.fitness_bleu(src, src_embeddings, new_solutions, pseudo_refs, remove_identical_pseudorefs)
            else:
                bleus=[0.0]*len(new_solutions)
            if self.chrf_w!=0:
                chrfs=self.fitness_chrf(src, src_embeddings, new_solutions, pseudo_refs, remove_identical_pseudorefs)
            else:
                chrfs=[0.0]*len(new_solutions)

            if self.comet_mbr_w!=0:
                start = timer()
                comet=self.fitness_comet_mbr(src, src_embeddings, new_solutions, pseudo_refs=pseudo_refs, model=model, pseudo_ref_embeddings=pseudo_ref_embeddings)
                end = timer()
                logger.info("Computing COMET  scores took {} s".format(end - start))
            else:
                comet=[0.0]*len(new_solutions)
            if self.comet_qe_w!=0:
                start = timer()

                comet_qe=self.fitness_comet_qe_new(src, new_solutions, model_qe)
                end = timer()
                logger.info("Computing COMET-QE  scores took {} s".format(end - start))
            else:
                comet_qe=[0.0]*len(new_solutions)
            if self.unite_w!=0:
                start = timer()

                unite=self.fitness_unite_ref(src, src_embeddings, new_solutions, pseudo_refs, self.unite_model)
                end = timer()
                logger.info("Computing UniTE  scores took {} s".format(end - start))
            else:
                unite=[0.0]*len(new_solutions)
            if self.bleurt_w!=0:
                start = timer()
                bleurt=self.fitness_bleurt_ref(src, src_embeddings, new_solutions, pseudo_refs, self.model_bleurt)
                end = timer()
                logger.info("Computing BLEURT scores took {} s".format(end - start))
            else:
                bleurt=[0.0]*len(new_solutions)

            total_scores = (((np.asarray(bleus) * self.bleu_w) / 100) + np.asarray(comet_qe)*self.comet_qe_w + (
                        (np.asarray(chrfs) * self.chrf_w) / 100) + np.asarray(comet)*self.comet_mbr_w +
                            np.asarray(unite)*self.unite_w + np.asarray(bleurt)*self.bleurt_w)
        else:
            total_scores = []
        computed_i = 0
        cache_i = 0
        final_scores = []
        if caching_scores:
            for i in range(len(solutions)):
                if i in cache_idx:
                    solutions[i].score = cached_scores[cache_i]
                    final_scores.append(cached_scores[cache_i])
                    cache_i += 1
                else:
                    solutions[i].score = total_scores[computed_i]
                    final_scores.append(total_scores[computed_i])
                    # self.score_cache[solutionsd[i]] = total_scores[computed_i]
                    computed_i += 1
        else:
            final_scores = total_scores
        return final_scores

    def fitness_comet_mbr_and_qe_and_bleu_and_chrf_w_multiref(self, src, src_embeddings, solutions, model=None, model_qe=None, pseudo_refs=None,
                                 pseudo_ref_embeddings=None, caching=True, caching_scores=True, remove_identical_pseudorefs=True, bleu_w=1, chrf_w=1, comet_mbr_w=1, comet_qe_w=1):
        #bleus=self.fitness_bleu_multiref(src, src_embeddings, solutions, pseudo_refs, remove_identical_pseudorefs)
        #solutionsd = [solution.chromosome_detok for solution in solutions if solution.chromosome_detok != '']

        cache_idx = []
        cached_scores = []
        new_solutions = []
        if caching_scores:
            for i, s in enumerate(solutions):
                #if s in self.score_cache and caching:
                if s.score is not None and caching:
                    cache_idx.append(i)
                    #cached_scores.append(self.score_cache[s])
                    cached_scores.append(s.score)
                else:
                    new_solutions.append(s)
        else:
            new_solutions = solutions

        logger.info(f"\n{len(new_solutions)}/{len(solutions)} of solutions are new.")
        if new_solutions:
            if self.bleu_w!=0:
                bleus=self.fitness_bleu_multiref(src, src_embeddings, new_solutions, pseudo_refs, remove_identical_pseudorefs)
            else:
                bleus=[0.0]*len(new_solutions)
            if self.chrf_w!=0:
                chrfs=self.fitness_chrf_multiref(src, src_embeddings, new_solutions, pseudo_refs, remove_identical_pseudorefs)
            else:
                chrfs=[0.0]*len(new_solutions)

            if self.comet_mbr_w!=0 and self.comet_qe_w!=0:
                mbr_and_qe=self.fitness_comet_mbr_and_qe(src, src_embeddings, new_solutions, model, model_qe, pseudo_refs,
                                     pseudo_ref_embeddings, caching, caching_scores, remove_identical_pseudorefs,comet_mbr_w=comet_mbr_w, comet_qe_w=comet_qe_w)
            elif self.comet_mbr_w!=0 and self.comet_qe_w==0:
                mbr_and_qe=self.fitness_comet_mbr(src, src_embeddings, new_solutions, model, model_qe, pseudo_refs,
                                     pseudo_ref_embeddings, caching, caching_scores, remove_identical_pseudorefs)
            elif self.comet_qe_w!=0 and self.comet_mbr_w==0:
                mbr_and_qe=self.fitness_comet_qe_new(src, new_solutions, model_qe, remove_identical_pseudorefs=remove_identical_pseudorefs)
            else:
                mbr_and_qe=[0.0]*len(new_solutions)
            total_scores = ((np.asarray(bleus)*bleu_w)/100)+np.asarray(mbr_and_qe)+((np.asarray(chrfs)*chrf_w)/100)
        else:
            total_scores = []
        computed_i = 0
        cache_i = 0
        final_scores = []
        if caching_scores:
            for i in range(len(solutions)):
                if i in cache_idx:
                    solutions[i].score=cached_scores[cache_i]
                    final_scores.append(cached_scores[cache_i])
                    cache_i += 1
                else:
                    solutions[i].score=total_scores[computed_i]
                    final_scores.append(total_scores[computed_i])
                    #self.score_cache[solutionsd[i]] = total_scores[computed_i]
                    computed_i += 1
        else:
            final_scores = total_scores
        return final_scores

    def fitness_comet_mbr_and_qe(self, src, src_embeddings, solutions, model=None, model_qe=None, pseudo_refs=None,
                                 pseudo_ref_embeddings=None, caching=True, caching_scores=True, remove_identical_pseudorefs=True, comet_qe_w=1, comet_mbr_w=1):
        start = timer()

        mbr_scores = self.fitness_comet_mbr(src, src_embeddings, solutions, model=model, model_qe=model_qe,
                                                pseudo_refs=pseudo_refs, pseudo_ref_embeddings=pseudo_ref_embeddings,
                                                caching=caching, remove_identical_pseudorefs=True)
        end = timer()
        logger.info("Computing MBR scores took {} s".format(end - start))
        start = timer()
        qe_scores = np.asarray(self.fitness_comet_qe_new(src, solutions, model_qe, remove_identical_pseudorefs=True))
        end = timer()

        total_scores = np.asarray(mbr_scores,dtype=qe_scores.dtype)*comet_mbr_w + qe_scores*comet_qe_w

        return total_scores

    def fitness_qe(self, src, src_embeddings, solutions, model=None, model_qe=None, pseudo_refs=None,
                                 pseudo_ref_embeddings=None, caching=True, caching_scores=True, remove_identical_pseudorefs=True):
        solutionsd = [solution.chromosome_detok for solution in solutions if solution.chromosome_detok != '']

        cache_idx = []
        cached_scores = []
        new_solutions = []
        if caching_scores:
            for i, s in enumerate(solutionsd):  # should also look at source, as only source-solution pair is unique
                if s in self.score_cache and caching:
                    cache_idx.append(i)
                    cached_scores.append(self.score_cache[s])
                else:
                    new_solutions.append(solutions[i])
        else:
            new_solutions = solutions

        if new_solutions:
            qe_scores = np.asarray(self.fitness_comet_qe(src, new_solutions, model_qe, remove_identical_pseudorefs=True))
            total_scores = qe_scores
        else:
            total_scores = []

        computed_i = 0
        cache_i = 0
        final_scores = []
        if caching_scores:
            for i in range(len(solutions)):
                if i in cache_idx:
                    final_scores.append(cached_scores[cache_i])
                    cache_i += 1
                else:
                    #      logger.warning("appending {}".format(src_embeddings[computed_i]))
                    final_scores.append(total_scores[computed_i])
                    self.score_cache[solutionsd[i]] = total_scores[computed_i]
                    computed_i += 1
        else:
            final_scores = total_scores

        return final_scores



    def mutation(self, solution, possible_tgt, deletions=True):
        # TODO: solve for multi-token expressions
        # It should be more probable to .replace existing word rather than an emtpy one (i.e. adding a new word)
        empty_repl = 0.1
        bitstring = solution.chromosome.copy()
        for i in range(len(bitstring)):
            # check for a mutation
            if rand() < (self.mutation_rate / len(bitstring)):
                if bitstring[i] == '':
                    if rand() > empty_repl:  # do not add word to an empty place
                        continue

                tgt = random.choice(possible_tgt)
                #          if rand() > empty_repl*(sum([len(t) for t in possible_tgt])/len(possible_tgt)): # deletion of a word or insertion of a new word should have the same probs, but also some possible tgts are longer than 1 tokent, so try we account for that
                # how does crossover factor into this?
                #               tgt = random.choice(possible_tgt)  # .split(' ')
                #   logger.info(
                #      "nonempty repl!!! {}".format(empty_repl * (sum([len(t) for t in possible_tgt]) / len(possible_tgt))))
                # else:
                #    logger.info("empty repl!!! {}".format(empty_repl*(sum([len(t) for t in possible_tgt])/len(possible_tgt))))
                #    tgt=''
                #                bitstring[i]=''
                # continue
                l = len(tgt)
                if rand() > empty_repl:  # *(sum([len(t) for t in possible_tgt])/len(possible_tgt)):
                    delete = False
                else:
                    delete = True
                    tgt = [''] * len(tgt)
                delete = False
                if l > 1:
                    # find empty places in the gene
                    space_i = [a for a, x in enumerate(bitstring) if x == '']
                    if l > len(space_i):  # wait, thats illegal (we cant make the gene longer)
                        continue
                    #            for x in range(l):
                    # print(bitstring)
                    #               print(space_i)
                    #              print(x)
                    #             print(space_i[-x-1])
                    #
                    #                   del bitstring[space_i[-x-1]]  #we need to iterate backwards to not mess up the order
                    #              print(i)
                    #             print(bitstring)
                    #                logger.warning("inserting {} instead of {}".format(tgt, bitstring[i]))
                    if bitstring[i]:
                        if i == 0 and bitstring[i][0].isupper():  # uppercase the first letter, if start of the sentence
                            tgt[0] = tgt[0].capitalize()
                    if delete:
                        bitstring[i] = ''
                    else:
                        bitstring[i] = tgt[0]
                    for x in range(1, l):
                        if delete:
                            bitstring.insert(i + x, '')
                        else:
                            bitstring.insert(i + x, tgt[x])
                        space_i = [a for a, t in enumerate(bitstring) if t == '']
                        # logger.warning(space_i)
                        # logger.warning(x)
                        # logger.warning(bitstring)
                        del bitstring[space_i[-1]]
                else:
                    x = 0
                    bitstring[i] = tgt[x]
        return bitstring

    def mutation_old(self, solution, possible_tgt, deletions=True):
        # TODO: solve for multi-token expressions
        # It should be more probable to .replace existing word rather than an emtpy one (i.e. adding a new word)
        empty_repl = 0.1
        del_prob=0.1
        bitstring = solution.chromosome.copy()
        for i in range(len(bitstring)):
            # check for a mutation
            if rand() < (self.mutation_rate / len(bitstring)):
                if bitstring[i] == '':
                    if rand() > empty_repl:  # do not add word to an empty place
                        continue

                tgt = random.choice(possible_tgt)
                #          if rand() > empty_repl*(sum([len(t) for t in possible_tgt])/len(possible_tgt)): # deletion of a word or insertion of a new word should have the same probs, but also some possible tgts are longer than 1 tokent, so try we account for that
                # how does crossover factor into this?
                #               tgt = random.choice(possible_tgt)  # .split(' ')
                #   logger.info(
                #      "nonempty repl!!! {}".format(empty_repl * (sum([len(t) for t in possible_tgt]) / len(possible_tgt))))
                # else:
                #    logger.info("empty repl!!! {}".format(empty_repl*(sum([len(t) for t in possible_tgt])/len(possible_tgt))))
                #    tgt=''
                #                bitstring[i]=''
                # continue
                l = len(tgt)
                if rand() < del_prob and deletions is True:  # *(sum([len(t) for t in possible_tgt])/len(possible_tgt)):
                    tgt = [''] * len(tgt)
                delete = False
                if l > 1:
                    # find empty places in the gene
                    space_i = [a for a, x in enumerate(bitstring) if x == '']
                    if l > len(space_i):  # wait, thats illegal (we cant make the gene longer)
                        continue
                    #            for x in range(l):
                    # print(bitstring)
                    #               print(space_i)
                    #              print(x)
                    #             print(space_i[-x-1])
                    #
                    #                   del bitstring[space_i[-x-1]]  #we need to iterate backwards to not mess up the order
                    #              print(i)
                    #             print(bitstring)
                    #                logger.warning("inserting {} instead of {}".format(tgt, bitstring[i]))
                    if bitstring[i]:
                        if i == 0 and bitstring[i][0].isupper():  # uppercase the first letter, if start of the sentence
                            tgt[0] = tgt[0].capitalize()
                    if delete:
                        bitstring[i] = ''
                    else:
                        bitstring[i] = tgt[0]
                    for x in range(1, l):
                        if delete:
                            bitstring.insert(i + x, '')
                        else:
                            bitstring.insert(i + x, tgt[x])
                        space_i = [a for a, t in enumerate(bitstring) if t == '']
                        # logger.warning(space_i)
                        # logger.warning(x)
                        # logger.warning(bitstring)
                        del bitstring[space_i[-1]]
                else:
                    x = 0
                    bitstring[i] = tgt[x]
        return bitstring

    def tournament_selection(self, pop, scores, k=3):
        selection_ix = randint(len(pop))
        for ix in randint(0, len(pop), k - 1):
            # check if better (e.g. perform a tournament)
            if scores[ix] > scores[selection_ix]:
                selection_ix = ix
        return pop[selection_ix]

    def roulette_wheel_selection(self, pop, scores):
        scores = np.exp(scores)
        s = sum(scores)
        selection_probs = [i / s for i in scores]
        return pop[np.random.choice(len(pop), p=selection_probs)]

    def selection(self, pop, scores, k=3):
        return self.tournament_selection(pop, scores, k)

    # crossover two parents to create two children
    def crossover(self, p1, p2):
        # children are copies of parents by default
        # c1, c2 = p1.copy(), p2.copy()
        c1, c2 = p1, p2
        if len(p1.chromosome)<3: return [c1, c2]
        # check for recombination
        if rand() < self.crossover_rate:

            # select crossover point that is not on the end of the string
            pt = randint(1, len(p1.chromosome) - 2)
            #        pt=2
            # perform crossover

            c1 = Solution(p1.chromosome[:pt] + p2.chromosome[pt:])
            c2 = Solution(p2.chromosome[:pt] + p1.chromosome[pt:])
            c1.original = False
            c1.score = None
            c2.original = False
            c2.score = None
            del p1
            del p2
        #     print("new crossover:")
        #    print(c1)
        #   print(c2)
        return [c1, c2]

    # genetic algorithm
    def run(self):
        # initial population of random bitstring
        # keep track of best solution

        logger.info("fitness function: {}".format(self.objective.__name__))
        pop = self.init_pop
        if self.model is not None:
            self.model.set_embedding_cache()
        if self.model_qe is not None:
            self.model_qe.set_embedding_cache()

        best, best_eval = pop[0], self.objective(self.src, self.src_embeddings, pop, model=self.model, model_qe=self.model_qe,
                                            pseudo_refs=self.pseudo_refs, pseudo_ref_embeddings=self.pseudo_ref_embeddings,
                                            caching=self.caching, caching_scores=self.caching_scores, remove_identical_pseudorefs=self.remove_identical_pseudorefs)[0]

        logger.warning("initial first: {}".format(best))
        logger.warning("initial first fitness: {}".format(best_eval))

        # logger.warning("tgt words: {}".format(possible_tgt))
        # enumerate generations
        # if objective.__name__ in max_scores:
        #    if best_eval>=max_scores[objective.__name__ ]:
        #       return [best,best_eval]
        self.log["iters"] = {}
        for gen in tqdm(range(self.generations), desc='generation'):
            # evaluate all candidates in the population
            #        scores = [objective(src,c,ref) for c in pop]

            # TODO: we dont remove same pseudo references here, since we don't know after the first iteration, if the solutions are from the initial ones or generated
            scores = self.objective(self.src, self.src_embeddings, pop, model=self.model, model_qe=self.model_qe,
                                            pseudo_refs=self.pseudo_refs, pseudo_ref_embeddings=self.pseudo_ref_embeddings,
                                            caching=self.caching, caching_scores=self.caching_scores, remove_identical_pseudorefs=self.remove_identical_pseudorefs)
            logger.warning("avg fitness: {}".format(sum(scores)))
            logger.warning("avg fitness: {}".format(len(scores)))

            logger.warning("avg fitness: {}".format(sum(scores) / len(scores)))

            # for p, s in zip(pop,scores):
            #    logger.warning("{} = {}".format(detok.detokenize(p).strip(),s))
            # check for new best solution

            # logger.warning("scores:{}".format(scores))

            gen_best = -9999
            gen_best_i = 0
            for i in range(self.n_pop):
                if scores[i] > gen_best:
                    gen_best = scores[i]
                    gen_best_i = i
                    if scores[i] > best_eval:
                        best, best_eval = pop[i], scores[i]
                        self.log["max"] = {"best_score": float(scores[i]), "best_sentence": pop[i].chromosome, "iter": gen}

                        logger.warning("Iteration %d: >%d, new best f(%s) = %.3f" % (
                        gen, gen, pop[i].chromosome, float(scores[i])))
            if self.objective.__name__ in max_scores:
                if best_eval >= max_scores[self.objective.__name__]:
                    return [best, best_eval]
            self.log["iters"][gen] = {"best_score": scores[gen_best_i], "avg_score": float(sum(scores) / len(scores)),
                                 "best_sentence": pop[gen_best_i].chromosome,"avg_len":len([place for p in pop for place in p.chromosome if place != '']) / len(pop)}
            if gen % 10 == 0:
                logger.warning("gen: {}".format(gen))
                logger.warning("pop sample: {}".format(list(
                    p.chromosome_detok.strip() + ' ' + str(s) for s, p in zip(scores[:50], pop[:50]))))



            selected = [self.selection(pop, scores) for _ in range(self.n_pop)]

            # logger.info ("after:")
            # for p in selected:
            #   logger.info (' '.join(p))
            # logger.info("Len select: {}".format(len(selected)))
            # create the next generation
            children = list()
            for i in range(0, self.n_pop, 2):
                # get selected parents in pairs
                p1, p2 = selected[i], selected[i + 1]
                # crossover and mutation
                for c in self.crossover(p1, p2):
                    # mutation
                    new_chromosome = self.mutation(c, self.possible_tgts,deletions=self.deletions)
                    if new_chromosome != c.chromosome:
                        del c
                        c = Solution(new_chromosome)
                        c.original = False
                        c.score = None
                    children.append(c)
            # replace population
            pop = children
            logger.warning("Average solution length: {}".format(len([place for p in pop for place in p.chromosome if place != '']) / len(pop)))
            # logger.info("Len pop: {}".format(len(pop)))

        return [best, best_eval]


def ga_init(cfg, model, model_qe, model_unite=None, model_bleurt=None):
    n_best = cfg.num_samples
    n_refs = cfg.num_pseudo_refs

    logger.info("CFG: {}".format(cfg))
    # with open("{}.trans".format(f)) as trans, open("{}.scores".format(f)) as scores, open(
    #       "{}.ref".format(f)) as refs, open("{}.src".format(f)) as srcs, open(
    #      "{}.tgt_words_exp".format(f)) as dict_tgts:
    with open(cfg.sources()) as srcsf, open(cfg.translations()) as trans, open(cfg.pseudo_ref()) as pseudo_refsf:
        translations = trans.readlines()
        pseudo_refs = pseudo_refsf.readlines()
        srcs = srcsf.readlines()
        i = 0
        if cfg.dict is not None:
            dict_tgts = open(cfg.dict())
        elif cfg.wordlist_dict is not None:
            dict_tgts = open(cfg.wordlist_dict())
        else:
            dict_tgts = [';'] * len(srcs)
        
        for dict_tgt, src in zip(dict_tgts, srcs):
            # We assume wordlist "dictionary" has only 1 line, same for each example, so we reset the reading for each one
            if cfg.wordlist_dict is not None:
                dict_tgts.seek(0)
            translations_sent = translations[i * n_best:(i + 1) * n_best]
            pseudo_refs_sent = pseudo_refs[i * n_refs:(i + 1) * n_refs]

            init_pop = ([tok.tokenize(s) for s in translations_sent])
            pop = []
            # refactor
            for s in init_pop:
                news = []
                for t in s:
                    news.append(t)
                    if cfg.no_empty == False:
                        news.append('')
                pop.append(news)
            tgt_toks = list(set([tok for sent in init_pop for tok in sent]))
            tgt_toks = [[tok] for tok in tgt_toks]

            # dict_toks = [t.split(' ') for t in dict_tgt.strip().split(';')] # for multitok
            if cfg.multitok_dict==False:
                dict_toks = [[tk] for t in dict_tgt.strip().split(';') for tk in tok.tokenize(t)]  # for singletok
            else:
                dict_toks = [['□'.join(tok.tokenize(t))] for t in dict_tgt.strip().split(';')] #multitok
            # possible_tgt = tgt_toks + [['']]*(len(tgt_toks)+len(dict_toks)) + dict_toks
            possible_tgt = tgt_toks + dict_toks  # + [[''] for tok in tgt_toks] + [['' for t in toks] for toks in dict_toks]

            #ATTENTION! NEW
           # possible_tgt= [tok for tok in possible_tgt if tok != ['']]
#            logger.info(possible_tgt)
            max_len = int(max([len(s) for s in pop]) * 1.1)
            # PAD
            if cfg.no_empty == True:
                pop=pop*50
            else:
                pop = [(p + max_len * [''])[:max_len] for p in pop] * 50
            orig_sents = translations_sent * 50  # Do not fuck up retokenization at least in first gen

            # logger.warning()
            # fitness=globals()[cfg.fitness]
            solution_pop = [Solution(s) for s in pop]
            for sol, orig in zip(solution_pop, orig_sents):
                sol.chromosome_detok = orig
            log = {}
            if cfg.no_empty == True:
                deletions = False
            else:
                deletions = True
            ga = GA(cfg.fitness, src, solution_pop, max_len, pseudo_refs_sent, possible_tgt, cfg, model=model, model_qe=model_qe,
                                 pseudo_refs=pseudo_refs_sent, model_unite=model_unite, model_bleurt=model_bleurt)
            print(ga.pseudo_ref_embeddings, file=sys.stderr)
            print(model,file=sys.stderr)
            out = ga.run()[0]
            print(out.chromosome_detok.strip().replace('□',' '))
            logger.warning(log)
            if cfg.logfile is not None:
                with open(cfg.logfile, 'a') as lf:
                    json.dump(ga.log, lf, indent=4)
            i += 1


def mbr_command() -> None:
    model = None
    model_qe = None
    parser = ArgumentParser(description="Command for Minimum Bayes Risk Decoding.")
    parser.add_argument("-s", "--sources", type=Path_fr)
    parser.add_argument("-t", "--translations", type=Path_fr)
    parser.add_argument("-m", "--mutation", type=float, default=1)
    parser.add_argument("-c", "--crossover", type=float, default=0.05)
    parser.add_argument("-g", "--generations", type=int, default=100)
    parser.add_argument("-p", "--pseudo_ref", type=Path_fr)
    parser.add_argument("--cache-embeddings", "--cache-embeddings", type=bool, default=True)
    parser.add_argument("--no-empty", "--no-empty", type=bool, default=False)

    parser.add_argument("--cache-scores", "--cache-scores", type=bool, default=True)
    parser.add_argument("--remove-identical-pseudorefs", "-remove-identical-pseudorefs", type=bool, default=True)
    parser.add_argument("--multitok-dict", type=bool, default=True)

    parser.add_argument("-d", "--dict", type=Path_fr, default=None)
    parser.add_argument("-w", "--wordlist_dict", type=Path_fr, default=None)
    parser.add_argument("--comet_mbr_batch_size", type=int, default=64)
    parser.add_argument("--qe_batch_size", type=int, default=64)

    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--num_pseudo_refs", type=int, required=True)
    parser.add_argument("-l", "--logfile", type=str, default=None)

    parser.add_argument("--bleu_w", type=float, default=1.0)
    parser.add_argument("--chrf_w", type=float, default=1.0)
    parser.add_argument("--comet_mbr_w", type=float, default=1.0)
    parser.add_argument("--comet_qe_w", type=float, default=1.0)
    parser.add_argument("--bleurt_w", type=float, default=1.0)
    parser.add_argument("--unite_w", type=float, default=1.0)

    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="Unbabel/wmt20-comet-da",
        help="COMET model to be used.",
    )
    parser.add_argument(
        "--model-qe",
        type=str,
        required=False,
        default="Unbabel/wmt20-comet-da-qe",
        help="COMET model to be used.",
    )
    parser.add_argument(
        "--model-bleurt",
        type=str,
        required=False,
        default="/lnet/work/people/jon/ga_clean_comet-v2/bleurt/checkpoints/BLEURT-20/",
        help="BLEURT checkpoint to be used.",
    )
    parser.add_argument(
        "--model_storage_path",
        help=(
                "Path to the directory where models will be stored. "
                + "By default its saved in ~/.cache/torch/unbabel_comet/"
        ),
        default=None,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="mbr_result.txt",
        help="Best candidates after running MBR decoding.",
    )
    parser.add_argument(
        "-f",
        "--fitness",
        type=str,
        default="fitness",
        help="The name of the fitness function.",
    )
    cfg = parser.parse_args()

    # TODO Load necessarry models based on fitness function specified by the user
    if cfg.sources is None:
        parser.error("You must specify a source (-s)")

    if cfg.model.endswith(".ckpt") and os.path.exists(cfg.model):
        model_path = cfg.model
    else:
        model_path = download_model(cfg.model, saving_directory=cfg.model_storage_path)
    if cfg.fitness not in ["fitness_bleu", "fitness_chrf", "fitness_bleu_multiref","fitness_chrf_multiref","fitness_qe"] and cfg.comet_mbr_w!=0:
        model = load_from_checkpoint(model_path)

        if not isinstance(model, RegressionMetric):
            raise Exception(
                "Incorrect model ({}). MBR command only works with RegressionMetric models.".format(
                    model.__class__.__name__
                )
            )

        model.eval()
        if torch.cuda.is_available():
            model.cuda()
        else:
            model.cpu()
            logger.warning("No CUDA available, using CPU instead.")

    if "qe" in cfg.fitness or "combined" in cfg.fitness and cfg.comet_qe_w!=0:
        model_qe = load_from_checkpoint(download_model(cfg.model_qe))
        model_qe.eval()
        if torch.cuda.is_available():
            model_qe.cuda()
        else:
            model_qe.cpu()
            logger.warning("No CUDA available, using CPU instead.")
    if cfg.bleurt_w!=0:
        model_bleurt = bleurt_score.BleurtScorer(cfg.model_bleurt)
    else:
        model_bleurt=None
    if cfg.unite_w!=0:
        model_unite = pipeline(task=Tasks.translation_evaluation,
                                model='damo/nlp_unite_mup_translation_evaluation_multilingual_base')
    else:
        model_unite = None
    ga_init(cfg, model, model_qe, model_unite, model_bleurt)


if __name__ == "__main__":
    mbr_command()
