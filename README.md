# nsmc-sentimental-analysis    

기술 개발 과제를 통해 연구된 워크로드 가속화를 위한 워크로드 단위 데이터 로더 모듈 (프로토타입)
    
    
## Requirements
    
review data(ratings.txt) : https://drive.google.com/file/d/1Q8McylLDQBweiV9zTkYUHBa-YLbxB7bg/view?usp=sharing    
nsmc model : https://drive.google.com/file/d/1vxuzXnPt7pWtaIGB20zSnHN_kmePPqn_/view?usp=sharing    
    
    
### How to use    
    
* main.py perfix 변수에 nsmc model path 기입    
* dataloader.py data_path 부분에 ratings.txt path 기입    
    
    
```
$ python main.py --{ use_baseloader | use_fastloader}
```


