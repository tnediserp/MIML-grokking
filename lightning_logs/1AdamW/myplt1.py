import matplotlib.pyplot as plt
alpha=[0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8]
validation_accuracy=[0.491563708,0.736857057,100,99.9999923706054,100,100,100,100,100,100,100,100,100]

plt.figure()
sizes = 50
plt.scatter(alpha,validation_accuracy,s=sizes)
plt.plot(alpha,validation_accuracy)
plt.xlabel('alpha')
plt.ylabel('best validation accuracy after 100000 steps')
plt.xlim(0,1)
plt.ylim(-5,105)
plt.title('AdamW,weight decay=0.1')

#plt.title('learning rate = {}'.format(learning_rate))
if not plt.savefig('plot1.png', bbox_inches='tight'):
    print("保存图形失败，请检查文件路径和权限。")
else:
    print("图形已成功保存。")
plt.show()