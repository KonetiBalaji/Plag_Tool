from flask import Flask, render_template, request
import re
from pygments import lex
from pygments.lexers import get_lexer_by_name
from pygments.token import Token
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import multiprocessing as mp
import itertools
import plotly.graph_objects as go
import networkx as nx
from fuzzywuzzy import fuzz
import os

app = Flask(__name__)

def preprocess_code(code, language):
    """Preprocess code by removing comments and standardizing tokens."""
    try:
        lexer = get_lexer_by_name(language)
        tokens = lex(code, lexer)
        # Filter out comments
        filtered_tokens = [t[1] for t in tokens if t[0] not in {Token.Comment.Single, Token.Comment.Multiline}]
        return ' '.join(filtered_tokens)
    except Exception as e:
        raise ValueError(f"Preprocessing failed: {e}")

def extract_control_structures(code):
    """Extract frequency of control structures for structural similarity."""
    return {
        'for': len(re.findall(r'\bfor\s+\w+.*:', code)),
        'while': len(re.findall(r'\bwhile\s+.*:', code)),
        'if': len(re.findall(r'\bif\s+.*:', code)),
        'switch': len(re.findall(r'\bswitch\s+.*:', code)),
        'do-while': len(re.findall(r'\bdo\s+.*while\s*\(.*\):', code))
    }

def compare_logical_structures(code1, code2):
    """Compare control structure similarity between two code snippets."""
    control1 = extract_control_structures(code1)
    control2 = extract_control_structures(code2)
    return fuzz.ratio(list(control1.values()), list(control2.values())) / 100

def compare_codes(code_pair, language):
    """Compare two code snippets and return similarity score."""
    code1, code2 = code_pair
    try:
        # Preprocess
        proc_code1 = preprocess_code(code1, language)
        proc_code2 = preprocess_code(code2, language)
        
        # Token-based similarity
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([proc_code1, proc_code2])
        token_similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

        # Logical structure similarity
        logical_similarity = compare_logical_structures(code1, code2)

        # Average the two similarities
        return (token_similarity + logical_similarity) / 2
    except Exception as e:
        return f"Error comparing codes: {e}"

def compare_multiple_codes(codes, language):
    """Compare all code pairs in parallel and return a similarity matrix."""
    if len(codes) < 2:
        return {"error": "At least two files are required for comparison"}
    
    # Generate unique pairs
    pairs = list(itertools.combinations(range(len(codes)), 2))
    code_pairs = [(codes[i], codes[j]) for i, j in pairs]
    
    # Parallelize comparisons
    with mp.Pool(processes=mp.cpu_count()) as pool:
        similarities = pool.starmap(compare_codes, [(pair, language) for pair in code_pairs])
    
    # Build similarity matrix
    similarity_matrix = {}
    for (i, j), similarity in zip(pairs, similarities):
        pair = (f"Code {i+1}", f"Code {j+1}")
        similarity_matrix[pair] = similarity * 100 if isinstance(similarity, float) else similarity
    return similarity_matrix

def plot_similarity_network(similarity_matrix, threshold):
    """Generate an interactive network graph with Plotly."""
    G = nx.Graph()
    for (code1, code2), similarity in similarity_matrix.items():
        if isinstance(similarity, float) and similarity >= threshold:
            G.add_edge(code1, code2, weight=similarity / 100)
    
    pos = nx.spring_layout(G)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    node_x, node_y = [pos[n][0] for n in G.nodes()], [pos[n][1] for n in G.nodes()]
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=list(G.nodes()), hoverinfo='text',
                            marker=dict(size=10, color='lightblue'))

    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
        title=f"Similarity Network (Threshold: {threshold}%)", showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    return fig.to_html(full_html=False)

def plot_heatmap(similarity_matrix):
    """Generate an interactive heatmap with Plotly."""
    codes = sorted(set([code for pair in similarity_matrix.keys() for code in pair]))
    n = len(codes)
    matrix = [[0] * n for _ in range(n)]
    code_index = {code: i for i, code in enumerate(codes)}
    
    for (code1, code2), similarity in similarity_matrix.items():
        if isinstance(similarity, float):
            i, j = code_index[code1], code_index[code2]
            matrix[i][j] = similarity
            matrix[j][i] = similarity  # Symmetric matrix

    fig = go.Figure(data=go.Heatmap(z=matrix, x=codes, y=codes, colorscale='Viridis'),
                    layout=go.Layout(title="Similarity Heatmap"))
    return fig.to_html(full_html=False)

@app.route('/')
def index():
    """Render the upload page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """Handle file uploads and display similarity results."""
    try:
        files = request.files.getlist('files[]')
        if not files or all(f.filename == '' for f in files):
            return render_template('index.html', error="No files uploaded")
        
        # Validate files (e.g., size < 1MB, text-based)
        codes = []
        for file in files:
            if file and file.filename.endswith(('.py', '.java', '.cpp', '.c')) and file.content_length < 1_000_000:
                codes.append(file.read().decode('utf-8', errors='ignore'))
            else:
                return render_template('index.html', error="Invalid file type or size exceeds 1MB")

        language = request.form.get('language', 'python')
        threshold = float(request.form.get('threshold', 70))

        similarity_matrix = compare_multiple_codes(codes, language)
        if "error" in similarity_matrix:
            return render_template('index.html', error=similarity_matrix["error"])

        network_html = plot_similarity_network(similarity_matrix, threshold)
        heatmap_html = plot_heatmap(similarity_matrix)
        
        return render_template('results.html', similarity_matrix=similarity_matrix,
                               network_html=network_html, heatmap_html=heatmap_html)
    except Exception as e:
        return render_template('index.html', error=f"An error occurred: {e}")

if __name__ == '__main__':
    app.run(debug=True)