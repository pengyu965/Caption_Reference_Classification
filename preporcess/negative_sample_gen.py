import os 
import json 
from lxml import etree 
import html
import nltk


def remove_postive(xml_path, ann_json_path, negative_txt_path):
    for xml_file in os.listdir(xml_path):
        xml_file_path = os.path.join(xml_path, xml_file)
        file_name, _ = os.path.splitext(xml_file)
        ann_json_file_path = os.path.join(ann_json_path, file_name+'.json')
        with open(xml_file_path, 'r') as xml:
            xml_content = xml.read()
            # xml_content = html.unescape(xml_content)
            with open(ann_json_file_path, 'r') as js:
                js_content = json.load(js)
        
        replace_idx = []
        for i in range(len(js_content)):
            for j in range(len(js_content[i]["Span"])):
                span_start = int(js_content[i]["Span"][j][0])
                span_end = int(js_content[i]["Span"][j][1])
                for k in range(span_start, span_end):
                    replace_idx.append(k)

        # print(replace_idx)

        if os.path.exists("./cache_data/negative_xmlfile/") == False:
            os.mkdir("./cache_data/negative_xmlfile/")
        with open("./cache_data/negative_xmlfile/"+file_name+".xml", 'w') as f:
            f.write(''.join([xml_content[i] if (i in replace_idx)==False else " " for i in range(len(xml_content)) ]))

        with open("./cache_data/negative_xmlfile/"+file_name+".xml", 'r', encoding = "utf-8", errors = "ignore") as f:
            notags = xml2txt(f)

        # print("a" in notags)

        with open(negative_txt_path+file_name+".txt", 'wb') as f:
            f.write(notags)
        
def xml2txt(file):
    parser = etree.XMLParser(recover=True)
    tree = etree.parse(file, parser=parser)
    notags = etree.tostring(tree, encoding = 'utf-8', method= 'text', xml_declaration=False)
    return notags

def negtive_json(negative_txt_path, negative_json_path):
    for txt_file in os.listdir(negative_txt_path):
        file_name, _ = os.path.splitext(txt_file)
        txt_file_path = os.path.join(negative_txt_path, txt_file)
        with open(txt_file_path, 'r') as f:
            sent_list = nltk.sent_tokenize(f.read())
            # print(sent_list)
        
        j_list = []
        for i in range(len(sent_list)):
            j_dic = {}
            j_dic["Text"] = sent_list[i]
            j_dic["Entity"] = "Normal"
            j_list.append(j_dic)

        with open(os.path.join(negative_json_path, file_name+'.json'), 'w') as f:
            f.write(json.dumps(j_list, indent = 4))
        
        # print(json.dumps(j_list, indent = 4))

        # break





if __name__ == "__main__":
    ann_path = './cache_data/annfile/'
    xml_path = './cache_data/xmlfile/'
    ann_json_path = './cache_data/ann_jsonfile/'
    negative_txt_path = './cache_data/negative_txtfile/'
    negative_json_path = './cache_data/neg_jsonfile/'

    if os.path.exists(negative_json_path) == False:
        os.mkdir(negative_json_path)
    if os.path.exists(negative_txt_path) == False:
        os.mkdir(negative_txt_path)

    remove_postive(xml_path, ann_json_path, negative_txt_path)
    negtive_json(negative_txt_path, negative_json_path)
