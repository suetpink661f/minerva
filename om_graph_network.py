import streamlit as st
import glob
import pandas as pd
import numpy as np
import altair as alt
import builtins as p
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
#import plotly.express as px
from streamlit_dynamic_filters import DynamicFilters
import warnings
warnings.filterwarnings('ignore')


st.set_page_config(
    page_title="OM Graph Network Dashboard",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")
alt.themes.enable("dark")

st.title('OM Graph Network Dashboard')
#st.markdown('Kamada Kawai Layout')

@st.cache_data
def load_data():
    #st.write('Function - Load Data')
    files = glob.glob("*om_graph_network_g.csv")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs,ignore_index=True)
    #df = pd.read_csv('2020_om_graph_network_g.csv')
    df = df[['from','to','amount','Year','Region','Product']].copy()
    return df

def load_anomalies():
    files_pos = glob.glob("*om_graph_network_with_pos.csv")
    df6s = [pd.read_csv(f) for f in files_pos]
    df6 = pd.concat(df6s,ignore_index=True)
    #df6 = pd.read_csv('2020_om_graph_network_with_pos2.csv')
    df6  = df6[['Name','anomalies','node_shape','Amount','Year','Region','Product','x','y']].copy()
    return df6

df = load_data()
df6 = load_anomalies()

dynamic_filters = DynamicFilters(df, filters=['Year','Region','Product'])

with st.sidebar:
    st.write("Apply filters in any order 👇")

dynamic_filters.display_filters(location='sidebar')
dynamic_filters.display_df()

new_df = dynamic_filters.filter_df()

st.button(label="Display Graph", key="btn_graph")

if st.session_state.get("btn_graph"):
    # Your code, including widgets if you want
    for theregion in dynamic_filters.filter_df()['Region'].unique():
        for theproduct in dynamic_filters.filter_df()['Product'].unique():
            df_filter=new_df[new_df['Region']==theregion].copy()
            df_filter=df_filter[df_filter['Product']==theproduct]

            fig = go.Figure()
            theyears = new_df['Year'].unique()
            #st.write(theyears)

            df_pos=df6[df6['Region']==theregion].copy()
            df_pos=df_pos[df_pos['Product']==theproduct]
            #df_pos = df6[(df6['Region']==regionlist)&(df6['Product']==theproduct)]
            df_pos_new = df_pos[['Name', 'x', 'y']]
            #df_pos_new = df_pos[['x', 'y']].to_numpy()
            x = df_pos_new.set_index('Name')['x'].to_dict()
            y = df_pos_new.set_index('Name')['y'].to_dict()

            ds = [x, y]
            d = {}
            for k in x.keys():
                d[k] = np.asarray(tuple(d[k] for d in ds))

            #st.write(d)

            for year in theyears:
                ##st.write('Start -',year, theregion, theproduct)
                df_year=df_filter[df_filter['Year'] == year].copy()
                if len(df_year)<20:
                    ##st.write('Skip -',year, theregion, theproduct)
                    array_to_list= theyears.tolist()
                    array_to_list.remove(year)
                    theyears = np.asarray(array_to_list)
                    continue
                G = nx.from_pandas_edgelist(df_year, source='from', target='to', edge_attr='amount')
                #st.write(G)
                edge_weights = nx.get_edge_attributes(G, 'amount')
                max_weight = p.max(edge_weights.values())
                min_weight = p.min(edge_weights.values())
                #st.write(max_weight)
                #st.write(min_weight)
                if min_weight==max_weight:
                    #print('Skip min_weight -',year, theregion, theproduct)
                    array_to_list= theyears.tolist()
                    array_to_list.remove(year)
                    theyears = np.asarray(array_to_list)
                    continue
                normalized_weights = {edge: ((weight - min_weight) / (max_weight - min_weight))+1.1 for edge, weight in edge_weights.items()}
                pos = d

                for node in G.nodes:
                    for neighbor in G.neighbors(node):
                        diff = pos[node] - pos[neighbor]
                        force = diff * normalized_weights.get((node, neighbor), 0)  # Use normalized weight as a factor
                        pos[neighbor] += force
                
                node_trace = go.Scatter(x=[], y=[], mode='markers', text=[], marker=dict(size=df_pos['Amount'].values, color=df_pos['anomalies'],symbol=df_pos['node_shape']))

                for node in G.nodes():
                    x, y = pos[node]
                    node_trace['x'] += tuple([x])
                    node_trace['y'] += tuple([y])
                    node_trace['text'] += tuple([f"{node}"])
                
                edge_trace = go.Scatter(x=[], y=[], line=dict(width=1,color='#888'))

                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_trace['x'] += tuple([x0, x1, None])
                    edge_trace['y'] += tuple([y0, y1, None])

                fig.add_trace(node_trace)
                fig.add_trace(edge_trace)
                
                ##st.write('End ',year, theregion, theproduct)
                
            if len(fig.data)<=0:
                print('Skip fig No Data -', theregion, theproduct)
                continue

            for i in range(0,len(fig.data)):
                fig.data[i].visible = False
            
            fig.data[0].visible = True
            fig.data[1].visible = True
                
            steps = []
            i2=0
           
            for i in range(0,len(theyears)):
                step = dict(
                    method="restyle",
                    args=["visible", [False,False]*len(theyears)],
                )
                step["args"][1][i+i2] = True
                step["args"][1][i+i2+1] = True
                steps.append(step)
                i2+=1
                
            sliders = [dict(
                active=0,
                currentvalue={"prefix": "Date: "},
                steps=steps
            )]
            
            fig.update_layout(
            #     autosize=False,
                sliders=sliders,
                title="Quantity, DPA per Unit and Semi-net Revenue"
            )
            
            fig['layout']['sliders'][0]['currentvalue']['prefix']='Date: '
            for i,date in enumerate(theyears):
                fig['layout']['sliders'][0]['steps'][i]['label']=str(date)
            
            st.plotly_chart(fig, use_container_width=True)
