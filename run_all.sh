for i in {000..163}
do
	echo $i
	sed "s/chunk=x/chunk=$i/" run_beam_samp_dict_mbr_comet0.4+qe0.4+bleu0.1+chrf0.1_multiref_comet22_qe22.sh > run_beam_samp_dict_mbr_comet0.4+qe0.4+bleu0.1+chrf0.1_multiref_comet22_qe22_chunk$i.sh
	sbatch run_beam_samp_dict_mbr_comet0.4+qe0.4+bleu0.1+chrf0.1_multiref_comet22_qe22_chunk$i.sh
done
