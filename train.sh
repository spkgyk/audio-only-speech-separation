###
 # @Author: Kai Li
 # @Date: 2022-05-27 11:38:30
 # @Email: lk21@mails.tsinghua.edu.cn
 # @LastEditTime: 2022-06-02 16:25:26
### 
file_list=("audio_train.py --conf_dir=configs/dprnn_lrs2.yml" "audio_train.py --conf_dir=configs/dprnn_wsj0.yml")

for py_file in "${file_list[@]}"
do
    python -B ${py_file}
done