import matplotlib.pyplot as plt

alpha0=[0.2,0.25,0.3,0.35,
       0.4,0.45,0.5,0.55,0.6,
       0.65,0.65,0.7,0.7,
       0.75,0.75,0.8,0.8]
validation_accuracy0=[0.890128910541534,0.963582277297973,1.09322810173034,0.93198162317276,
                     0.921169161796569,0.792270541191101,0.956429302692413,0.873878121376037,2.07226347923278,
                     5.83054971694946,3.43152141571044,26.2840938568115,5.80942249298095,
                     78.8265304565429,99.5323104858398,99.840591430664,98.2996826171875]
alpha=[0.2,0.25,0.3,0.35,
       0.4,0.45,0.5,0.55,0.6,
       0.65,0.7,
       0.75,0.8]
validation_accuracy=[0.890128910541534,0.963582277297973,1.09322810173034,0.93198162317276,
                     0.921169161796569,0.792270541191101,0.956429302692413,0.873878121376037,2.07226347923278,
                     (5.83054971694946+3.43152141571044)/2,(26.2840938568115+5.80942249298095)/2,
                     (78.8265304565429+99.5323104858398)/2,(99.840591430664+98.2996826171875)/2]

plt.figure()
sizes = 50
plt.scatter(alpha0,validation_accuracy0,s=sizes)
plt.plot(alpha,validation_accuracy)
plt.xlabel('alpha')
plt.ylabel('best validation accuracy after 100000 steps')
plt.xlim(0,1)
plt.ylim(-5,105)
plt.title('SGD_with_Nesterov_momentum,lr=0.01,weight decay=0')

#plt.title('learning rate = {}'.format(learning_rate))
if not plt.savefig('plot5.png', bbox_inches='tight'):
    print("保存图形失败，请检查文件路径和权限。")
else:
    print("图形已成功保存。")
plt.show()