import os
import ann2json
import shutil

def preprocess_1(root_path, ann_path, xml_path, json_path):
    
    for dataset in os.listdir(root_path):
        dataset_path = os.path.join(root_path, dataset)
        for document in os.listdir(dataset_path):
            document_path = os.path.join(dataset_path, document)
            with open(os.path.join(document_path, document+".xml.ann")) as ann:
                if (ann.read() == '') == False:
                    shutil.copy2(os.path.join(document_path, document+".xml.ann"), ann_path)
                    shutil.copy2(os.path.join(document_path, document+".xml.txt"), os.path.join(xml_path, document+".xml"))
                    j_file = ann2json.gen(os.path.join(document_path, document+".xml.ann"))
                    ann2json.writer(os.path.join(json_path, document+".json"), j_file)
            


if __name__ == "__main__":
    cache_data = "./cache_data/"
    original_data = "./original_data/"

    root_path = "./original_data/SCISUMM-DATA/"
    ann_path = './original_data/annfile/'
    xml_path = './original_data/xmlfile/'
    ann_json_path = './cache_data/ann_jsonfile/'

    if os.path.exists(cache_data) == False:
        os.mkdir(cache_data)
    if os.path.exists(ann_path) == False:
        os.mkdir(ann_path)
    if os.path.exists(xml_path) == False:
        os.mkdir(xml_path)
    if os.path.exists(ann_json_path ) == False:
        os.mkdir(ann_json_path)
    preprocess_1(root_path, ann_path, xml_path, ann_json_path)