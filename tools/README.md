To load some images of the data, run  
`$user cd MNIST`  
`$user python tools/load_data.py --path /path/to/data/folder --save /path/to/save/result/image`  

To load some samples of downsampling, run  
`$user python tools/extract_features.py --data_path /path/to/data/folder  
                                        --save_path /path/to/save/result/image   
                                        --type function using for downsampling (max | min | average)`  
