#new data
import random
import math
import sys, getopt

#Global vars
sInputData , sTestData , sFilename = "" , "" ,""
iLearningRate = 0.1
Ecpoch = 15000

#2D arrays
arrIndata = []
arrIndata2 = []

#LayerConnectionsHidden
arrConnectHidden = []
#TotalHiddenNodes
arrHiddenNodes = []
#arrConnectHidden2Out
arrConnectHidden2Out = []
#arrOutputNodes
arrOutputNodes = []
#Bias
arrBias1 = []

#Amount of lines
iLines = None
#layers
iHiddenLayers , iHiddenNodes , iOut = 0 , 0 , 0 


class Main():



    #Parameter constructor
    #this will initialize the default s
    def __init__(self,FN1 = None,FN2 = None):
        if FN1==None and FN2 == None:
            return

        ###################################
        ##       ReadDataFromFile        ##
        ###################################
        #reading data from file
        iLines = 0 
        iLines1 = 0 
        #extractData
        #input and output data stored in matrix
        f1 = FN1.readlines()
        iLines1 =len(f1)   
        for x in f1:
            arrIndata.append(x.replace("\n",""))

        ###################################
        ##         Matrix setups         ##
        ###################################

        #Initiialze defaults
        iHiddenLayers , iHiddenNodes , iOut = 1 , 4 , 3

        #LayerConnectionsHidden
        rows, cols = (iHiddenNodes, 4*(iHiddenLayers-1)+4) 
        arrConnectHidden = [[0 for col in range(cols)] for row in range(rows)]
        arrConnectHidden = self.RandomNumGenerator(arrConnectHidden)
        
        #TotalHiddenNodes
        rows, cols = (iHiddenNodes, iHiddenLayers+1) 
        arrHiddenNodes = [[0 for col in range(cols)] for row in range(rows)]


        #arrConnectHidden2Out
        rows, cols = (iHiddenNodes, 3) 
        arrConnectHidden2Out = [[0 for col in range(cols)] for row in range(rows)]
        arrConnectHidden2Out = self.RandomNumGenerator(arrConnectHidden2Out)

        #arrOutputNodes
        arrOutputNodes =[0 for i in range(3)]

        #Setup for BIAS
        arrBias1 = [[0 for col in range(iHiddenLayers)] for row in range(iHiddenNodes)]
        arrBias1 = self.RandomNumGenerator(arrBias1)
        arrBias2 = [0 for i in range(3)]
        arrBias2 = self.RandomNumGenerator(arrBias2,True)
        
        #Setup Start for hidden and start Nodes
        iNumber = 0 
        arrHiddenNodes = self.ExtarctStart(arrIndata,arrHiddenNodes,iNumber)
        sResult = arrIndata[iNumber].split(',')[4]
        sResult = self.SelectionProccess(sResult)

        ###################################
        ##     Starting with Training    ##
        ###################################
        iTotalCounter = 0 
        fTrainAccuracy = 0
        fCorrect = 0 
        icounter = -1
        iEpochCounter = 0 
        

        #fWrong = 0 
        while Ecpoch>iTotalCounter:#0.00009 && 0.0001
            icounter += 1
            iTotalCounter += 1
            if len(arrIndata)-1 < icounter:
                iEpochCounter += 1
                icounter =0
            arrHiddenNodes = self.ExtarctStart(arrIndata,arrHiddenNodes,icounter)
            sResult = arrIndata[icounter].split(',')[4]
            sResult = self.SelectionProccess(sResult)
            #StartingNetInput
            arrHiddenNodes = self.DetermineNetInput(arrConnectHidden,arrHiddenNodes,arrBias1)#ArrConnectHidden , arrHiddenNodes , arBias
            #StartingOutPut
            arrOutputNodes = self.DetermineOutput(arrHiddenNodes,arrBias2,arrConnectHidden2Out,arrOutputNodes)#arrHiddenLayer , arrBias1 , arrConnectHidden2Out , arrOutputNodes
            sResult

            #StartBacktrack
            #BackpropForHiddenLayer
            arrConnectHidden= self.BacktractHiddenToStart(arrHiddenNodes,arrConnectHidden,arrConnectHidden2Out,arrOutputNodes,sResult,arrBias1)#arrHiddenNodes,arrConnectHidden,arrConnectHidden2Out,arrOutputNodes,sResult,arrBias1
            #lastBacktrac
            arrConnectHidden2Out = self.BacktractOutToHidden(arrOutputNodes , arrConnectHidden2Out , sResult , arrHiddenNodes)#arrOutputNodes , arrConnectHidden2Out , #sResult , #arrHiddenNodes
            #BacktracBias
            arrBias2 = self.BacktracTobias(arrOutputNodes , arrBias2 , sResult , arrHiddenNodes)#arrOutputNodes , arrConnectHidden2Out , #sResult , #arrHiddenNodes
            fTotalError = self.CalcTotalError(sResult,arrOutputNodes)
            
            #Determine Accuracy
            sString = sResult.split(',')
            for i in range(0,3):
                sString[i] = float(sString[i])
            if arrOutputNodes.index(max(arrOutputNodes)) == sString.index(max(sString)):
                fCorrect +=1
            fTrainAccuracy = (fCorrect/(iTotalCounter))*100.0
            #if 
  
        

        ###################################
        ##     Starting with Testing    ##
        ###################################
        
 
        iTotalCounter = 0 
        fTestAccuracy = 0
        fCorrect = 0 
        icounter = -1
        iEpochCounter = 0 


        f1 = FN2.readlines()
        iLines =len(f1) 
        for x in f1:
            arrIndata2.append(x.replace("\n",""))

        while icounter<len(arrIndata2)-1:#0.00009 && 0.0001
            icounter += 1
            arrHiddenNodes = self.ExtarctStart(arrIndata2,arrHiddenNodes,icounter)
            sResult = arrIndata2[icounter].split(',')[4]
            sResult = self.SelectionProccess(sResult)
            #StartingNetInput
            arrHiddenNodes = self.DetermineNetInput(arrConnectHidden,arrHiddenNodes,arrBias1)#ArrConnectHidden , arrHiddenNodes , arBias
            #StartingOutPut
            arrOutputNodes = self.DetermineOutput(arrHiddenNodes,arrBias2,arrConnectHidden2Out,arrOutputNodes)#arrHiddenLayer , arrBias1 , arrConnectHidden2Out , arrOutputNodes
            sResult

            #Determine Accuracy
            sString = sResult.split(',')
            for i in range(0,3):
                sString[i] = float(sString[i])
            if arrOutputNodes.index(max(arrOutputNodes)) == sString.index(max(sString)):
                fCorrect +=1
            fTestAccuracy = (fCorrect/(icounter+1))*100.0


        ###################################
        ##      Output to text file      ##
        ###################################
        File_object = open(r"output.txt","w")
        #writes single line
        
        self.CompileString(arrConnectHidden,File_object)
        self.CompileString(arrConnectHidden2Out,File_object,False,"Output")
        self.CompileString(arrBias1,File_object,True)
        self.CompileString(arrBias2,File_object,True,"Output")
        File_object.write("\nEpoch value: "+str(Ecpoch)+"\n")
        File_object.write("\nTrain accuracy: "+ str(fTrainAccuracy)+"%\n")
        File_object.write("\nTest accuracy: "+ str(fTestAccuracy)+"%\n")
        

        File_object.close()
        
        FN1.close()
        FN2.close()
        return 
    def SelectionProccess(self,sIn): #determine what the output should be
        if sIn=="Iris-setosa":
            sIn="1,0,0"
        elif sIn=="Iris-versicolor":
            sIn="0,1,0"
        else:
            sIn="0,0,1"
        return sIn

    def ExtarctStart(self,arrInput,arrMainArray,Number=-1):#Input File array , ArrHidden , lineNumber
        if Number != -1:
            sString = arrInput[Number]
        else:
            sString = arrInput
        sString = sString.split(',')
        for j in range(0,len(arrMainArray)):
            arrMainArray[j][0] = sString[j]
            arrMainArray[j][1] = 0 
        return arrMainArray

    def RandomNumGenerator(self,Arr,Bool = False):
        if Bool==True:
            for j in range(0,len(Arr)):
                    Arr[j] = random.uniform(0, 0.5)
        else:
            for i in range(0,len(Arr)):
                for j in range(0,len(Arr[0])):
                    Arr[i][j] = random.uniform(0, 0.5)

        return Arr

    def DetermineNetInput(self,arrInLayer,arrHiddenLayer,arrBias1):#ArrConnectHidden , arrHiddenLayer , arBias
        fTotal = 0.0
        iHiddenNum = 0 ;
        for i in range(0,len(arrInLayer)):
            fTotal = 0
            for j in range(0,len(arrInLayer[0])):
                fHold  = float(arrHiddenLayer[j][iHiddenNum])*arrInLayer[j][i]
                fTotal = fTotal+ fHold
            fTotal = fTotal + (arrBias1[i][iHiddenNum])
            arrHiddenLayer[i][iHiddenNum+1] = self.DoEuler(fTotal)
            if i == 3:
                iHiddenNum = iHiddenNum+1
                



        return arrHiddenLayer

    def DetermineOutput(self,arrHiddenLayer,arrBias2,arrConnectHidden2Out,arrOutputNodes):#arrHiddenLayer , arrBias1 , arrConnectHidden2Out , arrOutputNodes
        fTotal = 0.0
        iHiddenNum = 0
        for i in range(0,len(arrConnectHidden2Out[0])):
            fTotal = 0 
            for j in range(0,len(arrConnectHidden2Out)):
                fTestHold = (arrHiddenLayer[j][len(arrHiddenLayer[0])-1])
                fHold  = float(arrConnectHidden2Out[j][i])*fTestHold
                fTotal = fTotal+ fHold
            fTotal = fTotal + arrBias2[i]
            arrOutputNodes[i] = self.DoEuler(fTotal)
  

        return arrOutputNodes

    def BacktractOutToHidden(self,arrOutputNodes,arrConnectHidden2Out,sResult,arrHiddenNodes): #arrOutputNodes , arrConnectHidden2Out , #sResult , #arrHiddenNodes
        fTotal = 0.0
        for i in range(0,len(arrConnectHidden2Out[0])):
            for j in range(0,len(arrConnectHidden2Out)):
                #startCalcOne
                #fTotal = -1*(float(sResult.replace(',','')[i])-arrOutputNodes[i])
                fTotal = (arrOutputNodes[i] - float(sResult.replace(',','')[i]))
                #Calc2
                fTotal = fTotal * (arrOutputNodes[i]*(1-arrOutputNodes[i]))
                #calc3
                #changed I to J
                fTotal = fTotal * (1*arrHiddenNodes[j][len(arrHiddenNodes[0])-1]) 
                iLest = iLearningRate
                #fLast =arrConnectHidden2Out[j][i]-(fTotal*iLearningRate)
                arrConnectHidden2Out[j][i] =arrConnectHidden2Out[j][i]-(fTotal*iLearningRate)
    
        return arrConnectHidden2Out

    def BacktracTobias(self,arrOutputNodes,arrBias2,sResult,arrHiddenNodes): #arrOutputNodes , arrConnectHidden2Out , #sResult , #arrHiddenNodes
        fTotal = 0.0
        for i in range(0,len(arrBias2)):
           #startCalcOne
            fTotal = -1*(float(sResult.replace(',','')[i])-float(arrOutputNodes[i]))
            #fTotal = (float(sResult.replace(',','')[i])-float(arrOutputNodes[i]))
            #Calc2
            fTotal = fTotal * (arrOutputNodes[i]*(1-arrOutputNodes[i]))
            #calc3
            fTotal = fTotal * (1*arrHiddenNodes[i][len(arrHiddenNodes[0])-1]) 
            #fLast =arrBias2[i]-(fTotal*iLearningRate)  
            arrBias2[i] = arrBias2[i]-(fTotal*iLearningRate)  

        return arrBias2
    #ArrConnectHidden , arrHiddenLayer , arBias
    def newHiddenLayers(self,arrOutputNodes , arrConnectHidden2Out , arrHiddenNodes,arrBias2):#arrHiddenNodes , arrConnectHidden2Out , arrOutputNodes
        fTotal = 0.0
        iHiddenNum = 0 ;
        arrNew = [0 for i in range(4)]
        for i in range(0,len(arrConnectHidden2Out)):
            for j in range(0,len(arrConnectHidden2Out[0])):
                fHold  = float(arrConnectHidden2Out[i][j])*arrOutputNodes[j]
                fTotal = fTotal+ fHold
            #arrHiddenNodes[i][len(arrHiddenNodes[0])-1] = fTotal
            arrNew[i] = fTotal
        return arrNew

    def CalcTotalError(self,sResult,arrOutputNodes):
        fTotalError = 0.0
        for i in range(0,len(arrOutputNodes)):
            fTotalError = fTotalError + (0.5*pow((float(sResult.replace(',','')[i])-arrOutputNodes[i]),2))
        return fTotalError
    
    def BacktractHiddenToStart(self,arrHiddenNodes,arrConnectHidden,arrConnectHidden2Out,arrOutputNodes,sResult,arrBias1):#arrHiddenNodes,arrConnectHidden,arrConnectHidden2Out,arrOutputNodes,sResult,arrBias1

        #bigForforhiddenNodes
        for i in range(0,len(arrHiddenNodes)):
            #getRightTotal aka C & D
            fBigTotal = 0.0
            fC = 0.0
            fD = arrHiddenNodes[i][len(arrHiddenNodes[0])-1]*(1-arrHiddenNodes[i][len(arrHiddenNodes[0])-1])
            for g in range(0,len(arrOutputNodes)):
                fcurrTotal = 0 
                fOut = arrOutputNodes[g]
                fcurrTotal = -1*(float(sResult.replace(',','')[g])-(fOut))
                #fcurrTotal = (float(sResult.replace(',','')[g])-(fOut))
                fcurrTotal = fcurrTotal * (fOut*(1-(fOut)))
                fcurrTotal = fcurrTotal * arrConnectHidden2Out[i][g]
                fC = fC + fcurrTotal
            for j in range(0,len(arrConnectHidden)):
                fBigTotal = 0.0
                fW = 0.0
                fI = 0.0
                fBigTotal = fC*fD
                fW = arrConnectHidden[j][i]
                fI = float(arrHiddenNodes[j][0])
                fBigTotal = fBigTotal*fI
                #fLast =arrConnectHidden[j][i]-(iLearningRate*fBigTotal) 
                arrConnectHidden[j][i] = arrConnectHidden[j][i]-(iLearningRate*fBigTotal) 
            #fLast = arrBias1[i][0] - (iLearningRate*(fC*fD*1))
            arrBias1[i][0] = arrBias1[i][0] - (iLearningRate*(fC*fD*1))
            #1-4MainNodesCalc
        return arrConnectHidden

    def DoEuler(self,inum):#converts num to euler value
        fOut = 1/(1+math.exp(-1*inum))
        return fOut

    def HiddenLayerCalc(self,iHiddenLayerAmount,ArrNodeFrom,ArrayFromLayers,ArrayTo,arrBias1,iLayerNum):
        iBias= arrBias1[iLayerNum-1]
        return
    
    def CompileString(self,arrHidden,FN,bBias=False,sI="Input",):
        if bBias==False:
            FN.write(sI+" Node 1\t\t"+sI+" Node 2\t\t"+sI+" Node 3\t\t"+sI+" Node 4\t\n")
            sHold = "" 
            for j in range(len(arrHidden[0])):
                for i in range(0,len(arrHidden)):
                    FN.write(str(arrHidden[i][j]) + "\t")
                FN.write("\n")
            FN.write("\n\n")
        else:
            if sI=="Input":
                FN.write("Bias 1 weight 1\t\tBias 1 weight 2\t\tBias 1 weight 3\t\tBias 1 weight 4\t\n")
            else:
                FN.write("Bias 2 weight 1\t\tBias 2 weight 2\t\tBias 2 weight 3\t\n")
            sHold = "" 
            for j in range(len(arrHidden)):
                try:
                    FN.write(str(arrHidden[j][0]) + "\t")
                except:
                    FN.write(str(arrHidden[j]) + "\t")
            FN.write("\n\n")
        return



#Main Operation

#Creating Main OBJ
#training set
#trainData = input("")

#first read aka testdata
#FN1  = open("Actual.txt", "r") 
#FN1 = open(trainData, "r") 

#Second input
#testset
#testData = input("")
#FN2  = open("Test.txt", "r") 
#

argumentList = sys.argv 
FN1 = open(sys.argv[1], "r") 
FN2 = open(sys.argv[2], "r") 


obj = Main(FN1,FN2)