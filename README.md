# AI_Project
Classifying Laptops with Keras

# Usage
Running image scraper (Set label number according to list of computers.txt) after which move some images to /test_images folder with same folder name as in /images
```
py image_scrapper.py "Macbook" --count 400 --label "4"
```
Running image classifier
```
py image_classifier.py
```
Running web app at localhost:8080 that uses generated model
```
py '/Web App/app.py'
```
