# ir-hand
This project implements the codes for the paper '3D Hand Pose Estimation with a Single Infrared Camera via Domain Transfer Learning' published in ISMAR'20.

# Install python dependencies
`conda env create -f environment.yml`

# Pretrained models
You can download the pretrained models in the following link:
https://www.dropbox.com/sh/54pmyizj95636zz/AAA-Y-gIQclrSK61gjuqgr3Ha?dl=0

# Datasets
You can download the datasets in the following link:
https://www.dropbox.com/sh/9dlvpb2vm57moj1/AADGXiwvfFfhoMdnkeDQsiQJa?dl=0

# dataset test
You can test it on our dataset by launching:
`python Demo/runDemo_dataset.py`

# real-time test
You can test it using SR300 realsense camera in real-time by launching:
`python Demo/runDemo_realtime.py`

# Citations
If you think this code is useful for your research, consider citing:
```
@INPROCEEDINGS{ismar20_gypark,
  title     = {3D Hand Pose Estimation with a Single Infrared Camera via Domain Transfer Learning},
  author    = {Park, Gabyong and Kim, Tae-Kyun and Woo, Woontack},
  booktitle = {ISMAR},
  year      = {2020}
}
```
# Acknowledgements
This work was partly supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.2019-0-01270, WISE AR UI/UX Platform Development for Smartglasses) and Next-Generation Information Computing Development Program through the National Research Foundation of Korea(NRF) funded by the Ministry of Science, ICT (NRF-2017M3C4A7066316).

