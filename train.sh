###
 # @Author: Kai Li
 # @Date: 2022-05-27 11:38:30
 # @Email: lk21@mails.tsinghua.edu.cn
 # @LastEditTime: 2022-06-02 16:25:26
### 
file_list=(            
            "audio_train.py --conf-dir=configs/dptnet_lrs2.yml"
            "audio_train.py --conf-dir=configs/dptnet_lrs2_unfolded.yml"
            )

for py_file in "${file_list[@]}"
do
    python -B ${py_file}
done