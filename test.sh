###
 # @Author: Kai Li
 # @Date: 2022-05-27 11:38:30
 # @Email: lk21@mails.tsinghua.edu.cn
 # @LastEditTime: 2022-06-02 16:25:26
### 
file_list=(
            "audio_test.py --conf-dir=Experiments/checkpoint/bsrnn_wsj0_2mix_8gpu_256_64_6/conf.yml" 
            "audio_test.py --conf-dir=Experiments/checkpoint/bsrnn_wsj0_2mix_8gpu_2048_512_6/conf.yml" 
            )

for py_file in "${file_list[@]}"
do
    python -B ${py_file}
done