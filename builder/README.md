# Create the data splits and the annotation files

-get the dataset audio from:[https://datacollective.mozillafoundation.org/datasets/cmflnuzw71qkz8x3kil3tgjvk]
- get the textGrid from:
  https://huggingface.co/datasets/pacscilab/VoxCommunis/tree/main/textgrids
- test languages:
-   Japanese(jp), espagnole?(es),Mandarin (zh-CN), basque(eu)
-   To-add: Germain,English, frensh
  
- dev languages:
- Turkish, ukranian, Tamil, thai

  1) Convert to wav and reasmple audio files
  2) Prepare dataset splits based on the validate file and sets from cm
  3) Merge all the output files to have a best distribution
  4) order speaker based on speech duration
  5) split sets balanced with dev,test 2hours each and all the reste to train
  6) align files
  7) correct phones
  8) phonebase item files
  9) triphone item files
  10) clean and correct item files
  11) copy audio files for deva nd test
  12) cereate  subfolders and copy audio files for train set
