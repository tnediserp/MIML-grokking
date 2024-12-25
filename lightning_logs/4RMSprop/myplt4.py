import matplotlib.pyplot as plt
alpha=[0.2,0.25,0.3,0.35,
       0.4,0.45,0.5,0.55,
       0.6,0.65,0.7,0.75,0.8]
validation_accuracy=[1.35512161254882,1.09111523628234,2.52049803733825,1.06278610229492,
                     2.28520798683166,5.29468584060668,97.6408081054687,99.8346710205078,
                     99.9468612670898,99.9696350097656,100,99.8299331665039,100]

plt.figure()
sizes = 50
plt.scatter(alpha,validation_accuracy,s=sizes)
plt.plot(alpha,validation_accuracy)
plt.xlabel('alpha')
plt.ylabel('best validation accuracy after 100000 steps')
plt.xlim(0,1)
plt.ylim(-5,105)
plt.title('RMSprop,RMSprop_alpha=0.99,dropout=0.1')

#plt.title('learning rate = {}'.format(learning_rate))
if not plt.savefig('plot4.png', bbox_inches='tight'):
    print("保存图形失败，请检查文件路径和权限。")
else:
    print("图形已成功保存。")
plt.show()