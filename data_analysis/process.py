import pandas as pd




def new_features(input_path,output_path):
    
    train = pd.read_csv(input_path)
    # Calculate width and height of bounding boxes
    train['width'] = train['x2'] - train['x1']
    train['height'] = train['y2'] - train['y1']

    # Calculate the area of the object
    train['object_area'] = (train['x2'] - train['x1']) * (train['y2'] - train['y1'])

    # Calculate the area of the image (assuming all images have the same size)
    image_width = 1280  # Example image width
    image_height = 720  # Example image height
    image_area = image_width * image_height

    # Calculate the ratio of the area of the object to the area of the image
    train['area_ratio'] = train['object_area'] / image_area

    # Calculate the aspect ratio for each image
    train['aspect_ratio'] = train['width'] / train['height']
    
    train.to_csv(output_path,index=False)
    



input_path = r'csv_data/train_output.csv'
output_path = r'csv_data/trainP.csv'    
new_features(input_path,output_path)



input_path = r'csv_data/val_output.csv'
output_path = r'csv_data/valP.csv'    
new_features(input_path,output_path)
