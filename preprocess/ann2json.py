import os 
import json
import re
import html 

def gen(path_file):
    with open(path_file, 'r', encoding="utf8", errors='ignore') as f:
        line = f.readline()
        j_file = []
        dic = {}
        while(line != ''):
            ## Next two removing things could be removed
            # Remove html tag
            line = re.sub(r'<[^>]+>', '', line)
            # Remove HTML special character like &quot;
            line = html.unescape(line)
            if '&quot;' in line:
                print(path_file)
            # print(line)

            line_list = line.split()

            if line_list[0][0] == 'T':
                
                dic = {}
                dic['Code'] = line_list[0]
                dic['Entity'] = line_list[1]
                
                span = []
                subspan = []
                for i in range(2, len(line_list)):
                    if line_list[i].replace(';','').isdigit() == False:
                        idx = i
                        break
                    else:
                        if (";" in line_list[i]) == False:
                            if subspan == []:
                                subspan.append(line_list[i])
                            else: 
                                subspan.append(line_list[i])
                                span.append(subspan)
                        else:
                            subspan.append(line_list[i].split(";")[0])
                            span.append(subspan)
                            subspan = []
                            subspan.append(line_list[i].split(";")[1])
                # print(span)
                dic['Span'] = span

                dic['Text'] = " ".join(line_list[idx:])
            
            elif line_list[0][0] == 'A':
                if dic['Entity'] == "Caption":
                    if line_list[1] == "Type":
                        dic['Type'] = line_list[3]
                    elif line_list[1] == "Num":
                        dic['Num'] = line_list[3]
                elif dic['Entity'] == "Reference":
                    if line_list[1] == "RefType":
                        dic['RefType'] = line_list[3]
                    if line_list[1] == "Type":
                        dic['Type'] = line_list[3]
                    elif line_list[1] == "Num":
                        dic['Num'] = line_list[3]
                
            # line = '' 
            line = f.readline()

            if line == "" or line.split()[0][0] == "T" :
                j_file.append(dic)


        # print(json.dumps(j_file, indent=4))
        # print(len(j_file))
        
        # writer(path_file, des_path, j_file)
        return j_file


def writer(des_path, j_file):
    with open(des_path,'w') as f:
        ppj = json.dumps(j_file, indent=4)
        f.write(ppj)

# if __name__ == "__main__":
#     print("jjj")
#     gen("./C10-1045.xml.ann", "./")