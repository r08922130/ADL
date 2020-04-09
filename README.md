# ADL HW1
## Extractive Seq_Tag task 
train cmd : python3.7 main.py --train emb_dim(300) epochs(11) pretrained(pre/no) model(m1/m2)

I trained my model with BCEwithlogitloss and RMSProp optimizer which initial learning rate is 0.00001 and decrease the rate 0.85 when validation loss get higher than previous one. I trained 11 epochs on this model and get the score 0.1948 on rouge-1 evaluation. 

For plotting the distribution of relative locations, I use the matplotlib package. First I predict the extractive sentence, then stores their relative location to an array, for example, if the prediction is [1, 3] and the number of sentence is 10, then I add [0.1, 0.3] to the array. After finishing the prediction, I use the relative location array to plot the histogram with 25 bins and the range in (0, 1).

# For training abstractive tasks
train cmd : python3.7 src/main.py train batch_size(32) epoch(25) attention(A/F) layer_num(1)

## Abstractive Seq2Seq w/o Attention task
I trained my Seq2Seq w/o Attention model with CrossEntropyLoss and RMSProp optimizer which initial learning rate is 0.0001 and decrease the rate 0.5 when (validation loss - training loss) larger than the dynamic threshold. I trained 11 epochs on this model and get the score 0.2177 on rouge-1 evaluation. 

## Abstractive Seq2Seq with Attention task
I trained my Seq2Seq with Attention model with CrossEntropyLoss and RMSProp optimizer which initial learning rate is 0.0001 and decrease the rate 0.7 when (validation loss - training loss) larger than the dynamic threshold. I trained 24 epochs on this model and get the score 0.2570 on rouge-1 evaluation. 

For plotting the Attention weight, I choose the short sentence which is clear to know in the valid data as the input, then I predict the output and store the Attention weight for each output word. I only keep previous 30 output words as the output, so we have 30 Attention weights at most. The brighter grid in the Attention Weight Figure means that the input has more important for predicting corresponding output word. 