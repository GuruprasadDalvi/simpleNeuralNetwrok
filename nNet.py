import random
import numpy as np
import json


#basic funcyions

def sigmoid(x):
    return 1/(1+np.exp(-x))
def d_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def tanH(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
def d_tanh(x):
    return 1-(tanH(x)**2)

def cost_function(y_true, y_pred):
    y_true=y_true.reshape((len(y_true),1))
    diff=(y_true-y_pred)
    diff=diff**2
    cost=sum(diff**2)
    return cost*100

class nNetwork:
    def __init__(self,layers,activeFun="sigmoid"):
        '''
            Layers: list of number of layes including input layer and output layer
            activeFun(Activation Function) : sigmoid(defalut),tanh
        '''
        self.layers=layers
        self.weights=[]
        self.activeFun=activeFun
        for i in range(len(layers)):
            if i!=0:
                h=self.layers[i-1]
                w=self.layers[i]
                we=np.random.randn(w,h)
                self.weights.append(we)
        
        self.baise=[np.random.rand(n, 1) for n in self.layers[1:]]


    def activate(self,x,deriv=False):
        if deriv:
            if self.activeFun=="sigmoid":
                return d_sigmoid(x)
            if self.activeFun=="tanh":
                return d_tanh(x)
        if self.activeFun=="sigmoid":
            return sigmoid(x)
        if self.activeFun=="tanh":
            return tanH(x)

    def feedforward(self,inputs):
        inputs=np.array(inputs)
        inputs=inputs.reshape((len(inputs),1))
        self.activisions=[inputs]
        self.z=[]
        for i in range(len(self.layers)-1):
            a=self.activisions[i]
            w=self.weights[i]
            b=self.baise[i]
            self.z.append(np.dot(w,a)+b)
            self.activisions.append(self.activate(np.dot(w,a)+b))

    def back_prop(self,expected):
        expected=expected.reshape((len(expected),1))

        #calculate all deltas
        err=expected - self.activisions[-1]
        delta_l=err*self.activate(self.z[-1],deriv=True)
        deltas=[0]*(len(self.layers)-1)
        deltas[-1]=delta_l
        for i in range(len(deltas)-2,-1,-1):
            delta=np.dot(self.weights[i+1].transpose(),deltas[i+1])*self.activate(self.z[i],deriv=True)
            deltas[i]=delta.reshape((len(delta),1))

        #change weights
        dw=[]
        db=[]
        deltas=[0]+deltas

        for i in range(1,len(self.layers)):
            t=self.activisions[i-1].transpose()
            dw_temp=np.dot(deltas[i],t)
            db_temp=deltas[i]
            dw.append(dw_temp)
            db.append(db_temp)
        
        for i in range(len(self.weights)):
            self.weights[i]+=dw[i]
        for i in range(len(self.baise)):
            self.baise[i]+=db[i]

    def mutate(self):
        mutation=[]
        self.weights=np.array(self.weights)
        for i in range(len(self.layers)):
            if i!=0:
                h=self.layers[i-1]
                w=self.layers[i]
                we=np.random.uniform(low=-1.0000,high=1.0000,size=(w,h))
                mutation.append(we)
        mutation=np.array(mutation)
        self.weights+=mutation
       


    def predict(self,inputs):
        inputs=np.array(inputs)
        inputs=inputs.reshape((len(inputs),1))
        activisions=[inputs]
        for i in range(len(self.layers)-1):
            a=activisions[i]
            w=self.weights[i]
            b=self.baise[i]
            d=np.dot(w,a)
            activisions.append(self.activate(d+b))
        #print((activisions[-1]))
        return activisions[-1]
    
    def save(self,acc):
        we={}
        counter=0
        for wei in self.weights:
            we[counter]=wei.tolist()
            counter+=1
        bai={}
        counter=0
        for b in self.baise:
            bai[counter]=b.tolist()
            counter+=1
        networkDic={
            "accuracy":acc,
            "layers":self.layers,
            "weights":we,
            "baises":bai
        }
        op_file=open("network.json","w")
        json.dump(networkDic,op_file,indent=3)
        op_file.close()

    def load(self,path):
        #loading alrady saved network
        with open(path,'r') as target:
            data= json.load(target)
        ws=[]
        bs=[]
        ls=data['layers']
        for w in data['weights']:
            l=data['weights'][w]
            ws.append(np.array(l))
        for b in data['baises']:
            l=data['baises'][b]
            bs.append(np.array(l))

        self.weights=ws
        self.layers=ls
        self.baise=bs
        print('Network Loaded Succsefully')
    def crossover(self,partner):
        child=nNetwork(self.layers,activeFun=self.activeFun)
        for i in range(len(self.layers)-1):
            if i%2==0:
                child.weights[i]=self.weights[i]
            else:
                child.weights[i]=partner.weights[i]
        return child
