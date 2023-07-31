import pandas as pd
import numpy as np
import networkx as nx
import io
import gzip
import os
import requests
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrow
from matplotlib.legend_handler import HandlerPatch
from adjustText import adjust_text


def read_zip(filepath):
    '''Reads a gzip file and returns a BytesIO object
    filepath is the path to the file'''
    with open(filepath,  'rb') as f:
        compressed_file = io.BytesIO(f.read())
    return compressed_file


def import_table(compressed_file, columns):
    '''Reads a gzip file and returns a pandas DataFrame
    compressed_file is a BytesIO object
    columns is a list of column names'''
    if not compressed_file:
        return pd.DataFrame(columns=columns)
    txt = gzip.GzipFile(fileobj=compressed_file)

    try:
        df = pd.read_table(txt, names=columns, quoting=3,
                           keep_default_na=False, delim_whitespace=True)
    except pd.errors.ParserError:
        df = pd.read_table(txt, names=columns, quoting=3,
                           keep_default_na=False, delim_whitespace=True,
                           engine='python')

    return df


def download_clickstream(startm, endm, fp):
    '''Downloads clickstream data from Wikimedia for the months between
    startm and endm. startm and endm should be strings in the format
    'YYYY-MM'. fp is the filepath to save the data to.'''

    for month in pd.date_range(startm, endm, freq='MS'):
        mp = month.strftime('%Y-%m')
        filename = 'clickstream-enwiki-' + mp + '.tsv.gz'
        filepath = '/'.join((fp, filename))
        if os.path.exists(filepath):
            continue
        print('downloading', filename)
        url = 'https://dumps.wikimedia.org/other/clickstream/%s/%s' %(mp,
                                                                      filename)
        r = requests.get(url, allow_redirects=True)
        with open(filepath, 'wb') as f:
            f.write(r.content)
    print('all files downloaded')

def get_abc_clickstream(startm, endm, fp):
    '''Reads clickstream data from Wikimedia for the months between
    startm and endm. startm and endm should be strings in the format
    'YYYY-MM'. fp is the filepath to read the data from. Returns a
    DataFrame with all edges to letters of the alphabet and a DataFrame
    with the aggregated data across all articles for different sources.'''

    exts = ['other-search', 'other-empty', 'other-internal', 'other-external',
            'other-other']
    type_dict = {'other-search': 'Search Engines',
                'other-empty': 'No Record',
                'other-internal': 'Wikimedia',
                'other-external': 'Rest of the Web',
                'other-other': 'Unknown',
                'other':'Wikipedia Other',
                'link':'Wikipedia Link'}

    dft = pd.DataFrame()
    dfg = pd.DataFrame(columns=['Search Engines', 'Rest of the Web',
                                'No Record', 'Unknown',
                                'Wikimedia', 'Wikipedia Link',
                                'Wikipedia Other'])

    for month in pd.date_range(startm, endm, freq='MS'): # loop through months
        mp = month.strftime('%Y-%m')
        filename = 'clickstream-enwiki-' + mp + '.tsv.gz'
        filepath = '/'.join((fp, filename))
        print('reading', filename)
        df = import_table(read_zip(filepath), ['source', 'target', 'type', 'n'])

        # create updated source type column
        df['type2'] = df['type'].copy()
        ixs = df[df['type2'] == 'external'].index
        df.loc[ixs, 'type2'] = df.loc[ixs, 'source']
        df['type2'] = df['type2'].map(type_dict)
        dfg.loc[month] = df.groupby('type2')['n'].sum() # save aggregated data

        # filter to only include edges to letters of the alphabet
        df = df[df['target'].isin(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))]
        df['month'] = month
        dft = pd.concat([dft, df], ignore_index=True) # save edges to letters
    
    print('Done reading, now sorting')
    dft = dft.sort_values(['month', 'target', 'n'],
                          ascending=[True, True, False])
    return dft, dfg

def plot_az_fig(edgelist, tsuffix=''):
    '''Plots a figure of the clickstream data for the letters of the
    alphabet. edgelist is a DataFrame with the data, tsuffix is a string
    to append to the title of the figure.'''

    # create networkx graph
    g = nx.from_pandas_edgelist(edgelist, create_using=nx.DiGraph(),
                                edge_attr=True)

    h = 10 # height in figure of the sources
    s = 5 # scaling factor for the nodes
    
    # specify node and label positions for letters
    nodepos = {x:(n*s, 0) for n, x in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}
    labpos = nodepos.copy()

    # specify node and label positions for sources
    nodepos.update({'Search Engines': (0.1*25*s, h),
                    'Rest of the Web': ((0.1 + 0.8/3)*25*s, h),
                    'No Record': ((0.1 + 0.8*2/3)*25*s, h),
                    'Unknown': (0.9*25*s, h),
                    'Wikimedia': (0.1*25*s, -h),
                    'Wikipedia Link': (25/2*s, -h),
                    'Wikipedia Other': (0.9*25*s, -h)})
    labpos.update({'Search Engines': (0.1*25*s, 1.05*h),
                    'Rest of the Web': ((0.1 + 0.8/3)*25*s, 1.05*h),
                    'No Record': ((0.1 + 0.8*2/3)*25*s, 1.05*h),
                    'Unknown': (0.9*25*s, 1.05*h),
                    'Wikimedia': (0.1*25*s, -1.05*h),
                    'Wikipedia Link': (25/2*s, -1.05*h),
                    'Wikipedia Other': (0.9*25*s, -1.05*h)})

    # specify colours (consistent with other plots)
    cmap = cm.Dark2(np.linspace(0, 1, 7))
    sources = ['Search Engines', 'Rest of the Web', 'No Record', 'Unknown',
               'Wikimedia', 'Wikipedia Link', 'Wikipedia Other']
    source_colours = {x: cmap[n] for n, x in enumerate(sources)}
    source_colours.update({x: np.array([0,0,0,1])
                           for n, x in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ')})
    edge_colours = [source_colours[x[0]] for x in g.edges()]
    
    # specify node sizes and edge widths
    # note the square root transformation of the edge weights for visualisation
    in_s = dict(g.in_degree(weight='weight'))
    ns = [x/2000 for x in  in_s.values()] 
    fs = {k: 10 if k in sources else v/90000 for k, v in in_s.items()}
    widths = [x**0.5/200 for x in nx.get_edge_attributes(g, 'weight').values()]

    # plot figure
    fig, ax = plt.subplots(figsize=(20, 10))

    nx.draw_networkx_nodes(g, nodepos, node_size=ns, ax=ax) # draw nodes

    # draw edges and customise arrows
    arrows = nx.draw_networkx_edges(g, nodepos, node_size=ns, width=widths,
                                    edge_color=edge_colours, arrows=True,
                                    arrowstyle='-|>,head_width=0.4,head_length=0.8',
                                    ax=ax)
    for a, w in zip(arrows, widths):
        a.set_mutation_scale(w+3)
        a.set_joinstyle('miter')
        a.set_capstyle('butt')
    
    # draw letter labels
    for lab in list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        nx.draw_networkx_labels(g, labpos, labels={lab:lab}, font_size=fs[lab],
                                font_color=source_colours[lab], ax=ax)
    # draw source labels with boxes
    for lab in sources:
        nx.draw_networkx_labels(g, labpos, labels={lab:lab}, font_size=fs[lab],
                                font_color=source_colours[lab],
                                bbox={'boxstyle':'Round', 'facecolor':'w',
                                      'pad':0.5,
                                      'edgecolor':source_colours[lab]},
                                ax=ax)
        
    # set plot limits and title
    ax.set_xlim(-0.05*s*25, 1.05*s*25)
    ax.set_ylim(-1.6*h, 1.6*h)
    ax.set_title('Traffic Sources for Wikipedia Articles about Letters of the Alphabet'+tsuffix)
    
    #### creating a custom weight arrow legend seemed very complicated?
    def make_legend_arrow(legend, orig_handle, xdescent, ydescent, width,
                          height, fontsize):
        weight = orig_handle._width
        p = FancyArrow(0, 0.5*height, 50, 0, weight, length_includes_head=False,
                       head_width=3+2*weight, head_length=3+1.5*weight)
        return p
    l1l = [FancyArrow(0, 0, 20, 0, width=(10**x)**0.5/200, lw=0, color='k')
           for x in range(2, 7)]
    l1 = ax.legend(l1l, [12*' '+ str(10**x) for x in range(2, 7)],
                   handler_map={FancyArrow :
                                HandlerPatch(patch_func=make_legend_arrow)},
                                ncols=5, loc='upper center',
                                title='Views from Source')
    ####

    # add legend for node sizes
    l2l = [Line2D([0], [0], color='k', lw=0, marker='o',
                  markersize=(x/2000)**0.5)
                  for x in range(500000, 3500000, 500000)]
    l2 = ax.legend(l2l, list(range(500000, 3500000, 500000)),
                   title='Total Article Page Views\n', ncols=6,
                   loc='lower center', handletextpad=1.5, borderpad=1.5)
    
    # re-add first legend to plot
    plt.gca().add_artist(l1)
    plt.savefig('figures/az_network%s.png' %tsuffix, dpi=300, bbox_inches='tight')
    plt.show()


def az_wikilinks_fig(edgelist, tsuffix=''):
    '''Plots a figure of the clickstream data from other Wikipedia artticles
    towards articles about the letters of the alphabet. The plot code is
    slightly simpler than the other figure, hence uglier plot. edgelist is a
    DataFrame, tsuffix is a string to append to the title of the figure.'''

    # create networkx graph
    g = nx.from_pandas_edgelist(edgelist, create_using=nx.DiGraph(),
                                edge_attr=True)

    h = 10 # height in figure of the sources
    s = 5 # scaling factor for the nodes

    # specify node and label positions
    o_arts = sorted([x for x in g.nodes()
                     if x not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'], reverse=True)
    nodepos = {}
    nodepos.update({x:(h, n*s) for n, x in enumerate(o_arts[:len(o_arts)//2])})
    nodepos.update({x:(-h, n*s) for n, x in enumerate(o_arts[len(o_arts)//2:])})
    nodepos.update({x:(0, n*s)
                    for n, x in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[::-1])})

    labpos = nodepos.copy()
    labpos.update({x:(h+0.2, n*s)
                   for n, x in enumerate(o_arts[:len(o_arts)//2])})
    labpos.update({x:(-h-0.2, n*s)
                   for n, x in enumerate(o_arts[len(o_arts)//2:])})    

    # specify node sizes based on in strength and edge widths 
    # note _no_ square root transformation of the edge weights for visualisation
    in_s = dict(g.in_degree(weight='weight'))
    ns = [x/20 for x in in_s.values()]
    widths = [x/1000 for x in nx.get_edge_attributes(g, 'weight').values()]
    # plot figure
    fig, ax = plt.subplots(figsize=(10, 20))

    # draw nodes
    nx.draw_networkx_nodes(g, nodepos, node_size=ns, ax=ax) # draw nodes

    # draw edges
    nx.draw_networkx_edges(g, nodepos, node_size=ns, 
                            arrows=True, width=widths, alpha=0.8,
                            arrowstyle='-|>,head_width=0.4,head_length=0.8',
                            ax=ax)
    
    # draw node labels (left, centre, right)
    nx.draw_networkx_labels(g, {x:labpos[x] for x in o_arts[:len(o_arts)//2]},
                            labels={x:x for x in o_arts[:len(o_arts)//2]},
                            font_size=8, horizontalalignment='left', ax=ax)    
    nx.draw_networkx_labels(g,
                            {x:labpos[x] for x in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'},
                            labels={x:x for x in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'},
                            font_size=8, ax=ax)
    nx.draw_networkx_labels(g, {x:labpos[x] for x in o_arts[len(o_arts)//2:]},
                            labels={x:x for x in o_arts[len(o_arts)//2:]},
                            font_size=8, horizontalalignment='right', ax=ax)
        
    # set plot limits and title
    ax.set_ylim(-0.05*s*25, 1.05*s*25)
    ax.set_xlim(-1.8*h, 1.8*h)
    ax.set_title('Traffic from Wikipedia Links towards Articles about Letters of the Alphabet'+tsuffix)

    plt.savefig('figures/az_wiki_network%s.png' %tsuffix, dpi=300, bbox_inches='tight')
    plt.show()
