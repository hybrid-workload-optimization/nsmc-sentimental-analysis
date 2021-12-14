# nsmc-sentimental-analysis    

기술 개발 과제를 통해 공개한 한국전자기술연구원 인공지능 영화 리뷰 감성분석 알고리즘    
    
    
## Requirements
    
review data(ratings.txt) : https://drive.google.com/file/d/1Q8McylLDQBweiV9zTkYUHBa-YLbxB7bg/view?usp=sharing    
nsmc model : https://drive.google.com/file/d/1vxuzXnPt7pWtaIGB20zSnHN_kmePPqn_/view?usp=sharing    
    
    
### How to use    
    
* main.py perfix 변수에 nsmc model path 기입    
* dataloader.py data_path 부분에 ratings.txt path 기입    
    
    
```
$ python main.py --{ use_baseloader | use_fastloader}
```


