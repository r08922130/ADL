# ADL HW1
## Extractive Seq_Tag task 
I trained my model with BCEwithlogitloss and RMSProp optimizer which initial learning rate is 0.00001 and decrease the rate 0.85 when validation loss get higher than previous one. I trained 11 epochs on this model and get the score 0.1948 on rouge-1 evaluation. 

For the figure 1 , which is the figure of this task, I recorded the training loss every 100 step in train stage and the validation loss every epoch (about 4500 step).

For the figure 2 , which is also the figure of this task but different model, I recorded the training loss every 100 step in train stage and the validation loss every epoch (about 4500 step).

## Abstractive Seq2Seq w/o Attention task
