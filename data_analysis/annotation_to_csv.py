import csv
import json
import configparser
import sys
import os
import argparse
from tqdm import tqdm

class DataConverter:
    def __init__(self, config_file='config.ini'):
        self.config = configparser.ConfigParser()
        try:
            self.config.read(config_file)
        except FileNotFoundError:
            print("config.ini not found. Please create one from config.ini.example")
            sys.exit(1)

    def convert_to_csv(self, type='train'):
        classes = ["pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle", "traffic light", "traffic sign"]
        
        if type == 'train':
            image_folder = os.path.join(self.config["paths"]["train_images_folder"])
            annotation_file = os.path.join(self.config["paths"]["train_json"])
        else:
            image_folder = os.path.join(self.config["paths"]["val_images_folder"])
            annotation_file = os.path.join(self.config["paths"]["val_json"])

        with open(annotation_file, 'r') as f:
            data = json.load(f)

        with open(f'{type}_output.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['image_name', 'image_path', 'Type', 'weather', 'scene', 'timeofday', 'category', 'trafficLightColor', 'occluded', 'truncated', 'manualShape', 'manualAttributes', 'x1', 'y1', 'x2', 'y2'])
            
            chunk_size = 1000
            for i in tqdm(range(0, len(data), chunk_size)):
                chunk = data[i:i+chunk_size]
                rows = []
                for item in chunk:
                    image_name = item['name']
                    image_path = os.path.abspath(os.path.join(image_folder, image_name))
                    Type = type
                    weather = item['attributes']['weather']
                    scene = item['attributes']['scene']
                    timeofday = item['attributes']['timeofday']
                    for label in item['labels']:
                        if label['category'] in classes:
                            category = label['category']
                            occluded = label['attributes'].get('occluded', 'N/A')
                            truncated = label['attributes'].get('truncated', 'N/A')
                            traffic_light_color = label['attributes'].get('trafficLightColor', 'N/A')
                            manualshape = label.get("manualShape", "N/A")
                            manualattributes = label.get("manualAttributes", "N/A")
                            if label.get('box2d'):
                                box2d = label.get('box2d', {})
                                x1 = box2d.get('x1', 'N/A')
                                y1 = box2d.get('y1', 'N/A')
                                x2 = box2d.get('x2', 'N/A')
                                y2 = box2d.get('y2', 'N/A')
                            else:
                                x1 = 'N/A'
                                y1 = 'N/A'
                                x2 = 'N/A'
                                y2 = 'N/A'
                            rows.append([image_name, image_path, Type, weather, scene, timeofday, category, traffic_light_color, occluded, truncated, manualshape, manualattributes, x1, y1, x2, y2])
                        else:
                            continue

                writer.writerows(rows)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert json to csv')
    parser.add_argument('--Type', type=str, default='train', help='Type of dataset')
    args = parser.parse_args()
    
    converter = DataConverter()
    converter.convert_to_csv(args.Type)
