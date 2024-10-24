# Sentiment_analysis
We use Bert pre-trained model and transformer structure to implement sentiment analysis.
## Bert VS GPT

- GPT is a transformer model that uses unidirectional attention. It can generate natural and fluent continuous text while understanding and encoding complex contextual information. 
- BERT (Bidirectional Encoder Representations from Transformers) is designed based on a bidirectional idea. It is trained into a deep bidirectional word representation method during the pre-training stage.


The advantage of GPT-2 over BERT is that it can generate natural language, so it is more suitable for tasks such as text generation; the advantage of BERT over GPT-2 is that it performs better in text classification, natural language inference, etc. So BERT is more suitable for sentiment analysis.

## Network structure
Word embedding (Bert pre-trained model) ->\\
Encoder (multi-head attention mechanism + position encoding) ->\\
TExtCNN ->\\
Decoder (multi-head attention mechanism) ->\\
softmax (output layer)
