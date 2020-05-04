import matplotlib.pyplot as plt

threshold = [0.95, 0.975,0.985,0.987,0.988,0.99]
#F1
ans_F1 = [0.8452,0.8446,0.8437,0.8391,0.8348,0.8150]
overall_F1 = [0.8467,0.8647,0.8743,0.8738,0.8724,0.8625]
unans_F1 = [0.8503,0.9119,0.9456,0.9549,0.9602,0.9735]
#EM
ans_EM = [0.7874,0.7868,0.7857,0.7812,0.7769,0.7579]
overall_EM = [0.8063,0.8243,0.8337,0.8333,0.8319,0.8226]
unans_EM = [0.8503,0.9119,0.9456,0.9549,0.9602,0.9735]

plt.figure(figsize=(4,6))

plt.ylim(0.75,1)
plt.xticks(threshold)
plt.plot(threshold,ans_F1,'ro-',label='answerable')
plt.plot(threshold,overall_F1,'bo-',label='overall')
plt.plot(threshold,unans_F1,'go-',label='unanswerable')
plt.legend()
plt.title("F1")
plt.savefig("Q6_F1.png")
plt.figure(figsize=(4,6))
plt.ylim(0.75,1)
plt.xticks(threshold)
plt.title("EM")
plt.plot(threshold,ans_EM,'ro-',threshold,overall_EM,'bo-',threshold,unans_EM,'go-')
plt.savefig("Q6_EM.png")