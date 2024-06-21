# WearSomethingToFaces
You can add something to face image, such as glasses, mask, etc.  
![Alt text](assets/show_case2.png)

If it's useful to you, please `star` ^_^
# Structure
```
-src: wear something class,glasses and mask.
-something_templetes: something templete imgs
```
You can add any class and templetes related you need,
wearing to faces.
# Usage
1. Set up enviroment
```bash
pip install -r requirements.txt
```
2. Inference
```
python main.py --path=path_to_dir --wear_type='glasses'
```
the result will be saved in `results` floder
# Thanks
the code is mainly refer to:
https://github.com/Amoswish/wear_mask_to_face

