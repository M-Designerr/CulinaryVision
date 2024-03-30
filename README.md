# IndianFood

model link - https://drive.google.com/file/d/1QztVDv6gWSUueEhEGKyomyx3WXDVAuFL/view?usp=drive_link

gcloud builds submit --tag gcr.io/culinaryvision/culinary_ai  --project=culinaryvision

gcloud run deploy --image gcr.io/culinaryvision/culinary_ai --platform managed  --project=culinaryvision --allow-unauthenticated --memory 2G