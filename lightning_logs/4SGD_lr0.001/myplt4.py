import matplotlib.pyplot as plt

alpha0=[0.2,0.25,0.3,0.35,
       0.4,0.45,0.5,0.55,0.6,
       0.65,0.65,0.7,0.7,
       0.75,0.75,0.8,0.8]
validation_accuracy0=[1.22226655483245,1.00609326362609,1.22988152503967,1.09548723697662,
                     0.956598758697509,0.792270541191101,0.998937249183654,1.08644306659698,4.17109441757202,
                     15.548131942749,0.85028851032257,51.611759185791,64.2933044433593,
                     26.5731296539306,72.4914932250976,77.7895812988281,95.9617385864257]
alpha=[0.2,0.25,0.3,0.35,
       0.4,0.45,0.5,0.55,0.6,
       0.65,0.7,
       0.75,0.8]
validation_accuracy=[1.22226655483245,1.00609326362609,1.22988152503967,1.09548723697662,
                     0.956598758697509,0.792270541191101,0.998937249183654,1.08644306659698,4.17109441757202,
                     (15.548131942749+0.85028851032257)/2,(51.611759185791+64.2933044433593)/2,
                     (26.5731296539306+72.4914932250976)/2,(77.7895812988281+95.9617385864257)/2]

plt.figure()
sizes = 50
plt.scatter(alpha0,validation_accuracy0,s=sizes)
plt.plot(alpha,validation_accuracy)
plt.xlabel('alpha')
plt.ylabel('best validation accuracy after 100000 steps')
plt.xlim(0,1)
plt.ylim(-5,105)
plt.title('SGD_with_Nesterov_momentum,lr=0.001,weight decay=0')

#plt.title('learning rate = {}'.format(learning_rate))
if not plt.savefig('plot4.png', bbox_inches='tight'):
    print("保存图形失败，请检查文件路径和权限。")
else:
    print("图形已成功保存。")
plt.show()