import urllib.request
from rdflib import *
import urllib.parse
import json
import time
import os

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "model")


def text2Wikifier(text, threshold, lang="zh", max_retries = 3):
    '''
    Calls wikifier on the text
    Input :
        text : input text to wikify
        threshold : pageRank threshold to use
        lang : language to use
    Output :
        Dictionary containing Wikifier's response
    '''
    # prepare the request
    data = urllib.parse.urlencode([
        ("text", text), ("lang", lang),
        ("userKey", "czgzhhslecgukacypvglbujzfmsdkm"),
        ("pageRankSqThreshold", "%g" % threshold), ("applyPageRankSqThreshold", "true"),
        ("nTopDfValuesToIgnore", "200"), ("nWordsToIgnoreFromList", "200"),
        ("wikiDataClasses", "true"), ("wikiDataClassIds", "false"),
        ("support", "false"), ("ranges", "false"), ("minLinkFrequency", "2"),
        ("includeCosines", "true"), ("maxMentionEntropy", "3")
        ])
    url = "http://www.wikifier.org/annotate-article"
    # call Wikifier and read the response.
    response = None
    last_error = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, data = data.encode("utf8"), method = "POST")
            with urllib.request.urlopen(req, timeout = 300) as f:
                response = f.read()
                response = json.loads(response.decode("utf8"))
            break
        except Exception as e:
            last_error = e
            print(f"  请求失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            time.sleep(3)  # 失败后等 3 秒再重试

    # 全部重试都失败：明确返回 None，避免后面 UnboundLocalError / TypeError
    if response is None:
        print(f"  Wikifier 调用彻底失败，放弃该条目: {last_error}")
        return None

    # return the Wikifier response if it contains a body (concepts)
    try :
        return response["annotations"]
    except KeyError :
        return None


def wikifier2RDF(data):
    '''
    Creates RDF graph from Wikifier response 
    Input :
        data : Wikifier response 
    Output :
        RDF graph
    '''
    # if no data is provided then exit
    if data == None:
        return None

    # create graph and add namespaces
    g = Graph()
    EX = Namespace("http://example.org/")
    SC = Namespace("http://schema.org/")
    UN = Namespace("https://univ-nantes.fr/ontology/")
    DCT = Namespace("http://purl.org/dc/terms/")
    WIKI = Namespace("http://en.wikipedia.org/wiki/")
    DBPR = Namespace("https://dbpedia.org/resource/")
    DBPT = Namespace("https://dbpedia.org/ontology/")
    g.bind("", EX)
    g.bind("sc", SC)
    g.bind("dct", DCT)
    g.bind("wiki", WIKI)
    g.bind("dbpr", DBPR)
    g.bind("dbpt", DBPT)    

    # adding triples to the graph
    for concept in data:
        # we can add other metadata or info here if we want
        g.add((URIRef("http://example.org/Chapter"), URIRef(DCT.subject), URIRef(concept["url"])))
        g.add((URIRef(concept["url"]), URIRef(DCT.title), Literal(concept["title"])))
        g.add((URIRef(concept["url"]), URIRef(UN.pageRank), Literal(concept["pageRank"])))
        #g.add((URIRef(concept["url"]), URIRef(UN.cosine), Literal(concept["cosine"])))
        '''if len(concept["wikiDataClasses"]) > 0 :
            for label in concept["wikiDataClasses"]:
                g.add((URIRef(concept["url"]), URIRef(RDFS.subClassOf), Literal("http://en.wikipedia.org/wiki/"+label["enLabel"])))
        '''
        if len(concept["dbPediaTypes"]) > 0 :
            for label in concept["dbPediaTypes"]:
                g.add((URIRef(concept["url"]), URIRef(RDFS.subClassOf), URIRef("https://dbpedia.org/ontology/"+label)))
    return g


def text2RDF(text, path, threshold = -1, lang="zh"):
    '''
    Creates RDF graph from text (pipeline)
    Input :
        text : input text to transform
        path : path in which the RDF graph will be saved
        threshold : pageRank threshold to use
    Output :
        RDF graph
    '''
    name = path.split("/")[-1].split(".")[0]
    print("Wikification for : " + name)
    # 只调用一次：之前这里重复调用了 text2Wikifier，第二次漏传 lang，
    # 导致同一段文本又被英文 Wikifier 处理一遍，返回英文实体、白等一次请求。
    data = text2Wikifier(text, threshold, lang=lang)

    # graph generation from wikifier results
    print("Graph generation for : " + name)
    g = wikifier2RDF(data)

    if g == None :
        print("There are no elements inside this chapter !")
        return None

    g.serialize(format = "turtle", destination = path.split(".")[0] + '.ttl')
    
    return g


def safe_text2RDF(row, version='v01', threshold=-1):
    cid = str(row['Cid']).replace(':', '_').replace('/', '_')
    output_path = os.path.join(DATA_ROOT, "output", "graphs", version, cid)     # ✅ 文件路径，不带扩展名
    ttl_file = output_path + ".ttl"

    if os.path.exists(ttl_file):
        print(f"跳过已存在: {ttl_file}")
        return None

    try:
        return text2RDF(row['Text'], output_path, threshold)
    except Exception as e:
        print(f"处理 {cid} 时发生异常: {e}")
        return None


