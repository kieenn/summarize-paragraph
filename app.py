from flask import Flask, url_for, render_template, request, redirect, jsonify
from controller import *
app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route('/')
def home():
    return render_template('summarize/index.html')

# @app.route('/summarize', methods=['POST'])
# def summarize():
#     # Get the text from the JSON data
#     text = request.get_json().get('text')
#     if text:
#         summary = summarize_text(text)
#         word_count = len(text.split())
#         sentence_count = text.count('.') + text.count('!') + text.count('?')  # Simple sentence count
#         return jsonify({'summary': summary, 'word_count': word_count, 'sentence_count': sentence_count})
#     else:
#         return jsonify({'error': 'Please provide text to summarize'}), 400
#
# @app.route('/summarize2', methods=['POST'])
# def summarize2():
#     text = request.get_json().get('text')
#     if text:
#         summary = summarize_text2(text)
#         word_count = len(text.split())
#         sentence_count = text.count('.') + text.count('!') + text.count('?')  # Simple sentence count
#         return jsonify({'summary': summary, 'word_count': word_count, 'sentence_count': sentence_count})
#     else:
#         return jsonify({'error': 'Please provide text to summarize'}), 400
#
# @app.route('/summarize3', methods=['POST'])
# def summarize3():
#     text = request.get_json().get('text')
#     if text:
#         summary = summarize_text3(text)
#         word_count = len(text.split())
#         sentence_count = text.count('.') + text.count('!') + text.count('?')  # Simple sentence count
#         return jsonify({'summary': summary, 'word_count': word_count, 'sentence_count': sentence_count})
#     else:
#         return jsonify({'error': 'Please provide text to summarize'}), 400

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.get_json().get('text')
    summary_length = int(request.get_json().get('summary_length'))
    if text:
        summary = summarize_with_kmeans(text=text, n_clusters=summary_length)
        return jsonify({'summary': summary})
    else:
        return jsonify({'error': 'Please provide text to summarize'}), 400
@app.route('/compare', methods=['POST'])
def compare():
    data = request.get_json()
    text1 = data.get('text1')
    text2 = data.get('text2')

    if text1 and text2:
        # Convert both texts to lowercase before comparison
        text1 = text1.lower()
        text2 = text2.lower()
        
        similarity = text_similarity(text1, text2)
        #similarity = calculate_text_similarity(text1, text2)

        return jsonify({
            'similarity': similarity
        })
    else:
        return jsonify({'error': 'Please provide both text1 and text2'}), 400

if __name__ == '__main__':
    app.run(debug=True)