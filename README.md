# Sentiment-Analysis

### Description: Sentiment analysis on IMDB dataset.using sequence to one models.
### Model Construction details:
<ul>
<li>Step-1 We make a LSTM layer and pass in the input vectors and at the last cell we store
the hidden state. </li>
<li>Step-2 Now we connect a dense layer at the end so as to map it to a review polarity between [0-1]</li>
</ul>

### Dependencies:
<ul>
<li>Vocab.py - Handles all the data loading , preprocessing and padding tasks.</li>
<li>Model.py - Handles the model along with its hyperparams.</li>
</ul>

### Outputs:
<ul>
<li>Example-1: "The movie was good at screenplay but the storyline was so boring and the actors were also not that good overall review would be a below average" - output: [[0.398776]]</li>
<li>Example-2: "A wonderful little production.The filming technique is very unassuming- very old-time-BBC fashion and gives a comforting, and sometimes discomforting, sense of realism to the entire piece.The actors are extremely well chosen- Michael Sheen not only ""has got all the polari"" but he has all the voices down pat too! You can truly see the seamless editing guided by the references to Williams' diary entries, not only is it well worth the watching but it is a terrificly written and performed piece. A masterful production about one of the great master's of comedy and his life.The realism really comes home with the little things: the fantasy of the guard which, rather than use the traditional 'dream' techniques remains solid then disappears. It plays on our knowledge and our senses, particularly with the scenes concerning Orton and Halliwell and the sets (particularly of their flat with Halliwell's murals decorating every surface) are terribly well done." - output:[[0.498843]]</li>
</ul>

### Improvements:
<ul>
<li>Attention mechanism must be added to improve accuracy as the sentence looses the context in large sentences as we saw in Example 2</li>
<li>Pretrained Embeddings such as Glove could be used to improve.</li>
<li>Stopwords,less frequent could be removed to improve accuracy.</li>
</ul>