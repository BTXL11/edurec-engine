import networkx as nx
import matplotlib.pyplot as plt
import rdflib
from pyvis.network import Network

def visualise_rdf_graph(input_path, output_path, format = 'turtle', to_gnx = True, to_gpv = True, hide_urls = False):
    '''
    Plot an RDF graph
    Input :
        input_path : graph input path
        output_path : plot file output path
        format : input graph type ex. : turtle, n3, ...
        to_gnx : visualize graph using networkx (static plot)
        to_gpv : visualize graph using pyvis (dynamic plot)
        hide_urls : hide namespace URLs
    Output :
        None
    '''
    
    # rdflib graph reading
    g = rdflib.Graph()
    g.parse(input_path, format = format)
    name = url_slicer(input_path).split(".")[0]

    # plot graph using networkx
    if to_gnx:
        gnx = nx.MultiDiGraph(name = name)
        for s, p, o in g:
            if hide_urls:
                s = url_slicer(s)
                p = url_slicer(p)
                o = url_slicer(o)
            gnx.add_edge(s, o, rel = p)
        visualise_graph_gnx(gnx, name, output_path)
    
    # plot graph using pyvis
    if to_gpv:
        gpv = Network('500px', '1800px', directed = True, select_menu = True, filter_menu = True)
        for s, p, o in g:
            if hide_urls:
                s = url_slicer(s)
                p = url_slicer(p)
                o = url_slicer(o)
            gpv.add_node(s)
            gpv.add_node(o)
            gpv.add_edge(s,o, title = p, color = 'green')
        visualise_graph_gpv(gpv, name, output_path)



def gnx2gpv(gnx):
    '''
    Transform Networkx graph to pyvis graph
    Input :
        gnx : Networkx input graph
    Output :
        Pyvis graph
    '''

    gpv = Network('500px', '1800px', directed = True, select_menu = True, filter_menu = True)
    for s, o, p in gnx.edges(data = True):
        type(s)
        gpv.add_node(str(s))
        gpv.add_node(str(o))
        gpv.add_edge(str(s), str(o), title = str(p), color = 'green')
    return gpv



def visualise_graph_gnx(gnx, name, path):
    '''
    Visualize Networkx graph
    Input :
        gnx : Networkx input graph
        name : Graph name to use when saving
        path : Path in which to save the graph plot
    Output :
        None
    '''
    edge_labels = dict([((n1, n2), d['rel']) for n1, n2, d in gnx.edges(data = True)])
    pos = nx.spring_layout(gnx)
    nx.draw(gnx, pos, with_labels = True, alpha = 0.5)
    nx.draw_networkx_edge_labels(gnx, pos, edge_labels = edge_labels, label_pos = 0.5, font_color = 'red')
    plt.savefig(path + name + '.png', format="PNG")
    plt.show()



def visualise_graph_gpv(gpv, name, path):
    '''
    Visualize Pyvis graph
    Input :
        gpv : Pyvis input graph
        name : Graph name to use when saving
        path : Path in which to save the graph plot
    Output :
        None
    '''
    gpv.toggle_physics(True)
    gpv.show_buttons(filter_= ['physics'])
    gpv.save_graph(path + name + '.html')



def url_slicer(url):
    '''
    Gets concept name from URL by removing namespace
    Input :
        url : object name with namespace
    Output :
        object name without namespace
    '''
    return url.split("/")[-1]