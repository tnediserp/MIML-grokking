import matplotlib.pyplot as plt
alpha=[0.2,0.25,0.3,0.35,
       0.4,0.45,0.5,0.55,
       0.6,0.65,0.7,0.75,0.8]
validation_accuracy=[1.14255344867706,1.07694494724273,0.986941993236541,1.01373445987701,
                     1.09831702709198,0.985507249832153,0.913921356201171,0.968351423740387,
                     1.19553661346435,1.21469783782958,0.991852641105651,0.850340127944946,0.850159406661987]

plt.figure()
sizes = 50
plt.scatter(alpha,validation_accuracy,s=sizes)
plt.plot(alpha,validation_accuracy)
plt.xlabel('alpha')
plt.ylabel('best validation accuracy after 100000 steps')
plt.xlim(0,1)
plt.ylim(-5,105)
plt.title('SGD_with_Nesterov_momentum,lr=0.001,weight decay=0.1')

#plt.title('learning rate = {}'.format(learning_rate))
if not plt.savefig('plot6.png', bbox_inches='tight'):
    print("保存图形失败，请检查文件路径和权限。")
else:
    print("图形已成功保存。")
plt.show()