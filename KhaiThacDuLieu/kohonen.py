import math
import pandas as pd
import streamlit as st

class SOM:
    def winner( self, weights, sample ) :
          
        D0 = 0       
        D1 = 0
        D2 = 0
          
        for i  in range( len( sample ) ) :
              
            D0 = D0 + math.pow( ( sample[i] - weights[0][i] ), 2 )
            D1 = D1 + math.pow( ( sample[i] - weights[1][i] ), 2 )
            D2 = D2 + math.pow( ( sample[i] - weights[2][i] ), 2 )

            if (D0 <= D1) and (D0 <= D2):
                return 0
            elif (D1 < D0) and (D1 <= D2 ): 
                return 1
            else :
              return 2

    # Function here updates the winning vector
    def update( self, weights, sample, J, alpha ) :
        i = 0
        #print("J: " + str(J)  + " " + str( len ( weights ) ) + " sample:" + str( len(sample)) )
        for i in range( len ( weights[J] ) ) :
          #print( "i: " + str(i) )
          weights[J][i] = weights[J][i] + alpha * ( sample[i] - weights[J][i] )  #wij = wij(old) - alpha(t) *  (xik - wij(old))
  
        return weights



# Driver code
def kohomen(data, agePara, distancePara) :
    age = data['Age']
    distance = data['Flight Distance']
    # Training Examples ( m, n )
    T = pd.concat([age, distance], axis=1).values
  
    m = len( T )
    n =  len( T[0] )
      
    # weight initialization ( n, C )
    #weights = [ [ 0.2, 0.6, 0.5, 0.9 ], [ 0.8, 0.4, 0.7, 0.3 ] ]
    weights = [ [ (age.values[1] + age.values[100] + 1)/2 , (distance.values[1] + distance.values[100] + 1)/2],
                [ (age.values[4] + age.values[400] + 1)/2 , (distance.values[4] + distance.values[400] + 1)/2],
                [ (age.values[5] + age.values[500] + 1)/2 , (distance.values[5] + distance.values[500] + 1)/2]
               ]

    # training
    ob = SOM()
      
    epochs = 5
    alpha = 0.5
      
    for i in range( epochs ) :
        for j in range( m ) :
              
            # training sample
            sample = T[j]
            
            # print(str(j) + " sample: " + str(sample))

            # Compute winner vector
            J = ob.winner( weights, sample )
            # print(str(j) + " J: " + str(J)) 
            # print(str(j) + " J weights: " + str(weights) )
            # Update winning vector
            weights = ob.update( weights, sample, J, alpha )
            # print(str(j) + " weights: " + str(weights) ) 
              
    # classify test sample
    s = [agePara,distancePara]
    J = ob.winner( weights, s )
    st.write(weights)
    return J


def Kmeans(centroids, sample) :  
  D0 = 0       
  D1 = 0
  D2 = 0
  st.write(centroids)
  for i  in range( len(sample)  ) :            
    D0 = D0 + math.pow( ( sample[i] - centroids[0][i] ), 2 )
    D1 = D1 + math.pow( ( sample[i] - centroids[1][i] ), 2 )
    D2 = D2 + math.pow( ( sample[i] - centroids[2][i] ), 2 )

  if (D0 <= D1) and (D0 <= D2):
    return 0
  elif (D1 < D0) and (D1 <= D2 ): 
    return 1
  else :
    return 2# classify test sample
