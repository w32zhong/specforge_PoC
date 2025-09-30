#!/bin/bash
if [ "$1" == "-h" ]; then
cat << USAGE
Description:
Upload checkpoints to huggingface.

Examples:
$0 <hgf_account> output/upbeat-bee-96/checkpoint-84510 ...
USAGE
exit
fi

[ $# -le 1 ] && echo 'bad arg.' && exit

hgf_account=${1}
shift 1

for ckpt_path in $@; do
	outdir=$(cd $ckpt_path/.. && pwd)
	model_name=$(basename $outdir)
	echo $outdir $model_name
	pushd $ckpt_path
	pwd
	set -xe
	hf upload $hgf_account/$model_name ../specforge_het.json
	hf upload --include config.json *.py --  $hgf_account/$model_name .
	hf upload $hgf_account/$model_name ./draft_model ./draft_model
	set +xe
	popd
done
