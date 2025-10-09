import numpy as np

def gradient_descent(x,y):
    if(len(x)!= len(y)):
        print("the two arrays must be of the same length")
        return 
    m_curr = b_curr = 0
    learning_rate=0.001
    iterations = 10000
    n= len(x)
    for i in range(iterations):
        y_predicted= m_curr * x + b_curr
        cost_function= (1/n)*np.sum((y-y_predicted)**2)
        m_derivative = -(2/n)*np.sum(x*(y-y_predicted))
        b_derivative = -(2/n)*np.sum(y-y_predicted)
        m_curr = m_curr- learning_rate* m_derivative
        b_curr = b_curr -learning_rate*b_derivative
        print(f"m : {m_curr}, b: {b_curr}, iteration: {i}, cost : {cost_function} ")


x= np.array([1,2,3,4,5])
y= np.array([5,7,9,11,13])

gradient_descent(x,y)

# learing rate of 0.001
# m : 2.450067366816869, b: 1.3751139996839103, iteration: 990, cost : 0.37751145148370463 
# m : 2.4499152007487943, b: 1.3756633674836414, iteration: 991, cost : 0.3772562242265526
# m : 2.449763086127419, b: 1.3762125495441813, iteration: 992, cost : 0.3770011695236341
# m : 2.4496110229353505, b: 1.3767615459283284, iteration: 993, cost : 0.3767462872582703
# m : 2.4494590111552026, b: 1.3773103566988596, iteration: 994, cost : 0.37649157731386695
# m : 2.449307050769595, b: 1.3778589819185307, iteration: 995, cost : 0.3762370395739047
# m : 2.449155141761153, b: 1.3784074216500761, iteration: 996, cost : 0.37598267392194695
# m : 2.449003284112507, b: 1.3789556759562092, iteration: 997, cost : 0.3757284802416274
# m : 2.448851477806295, b: 1.3795037448996217, iteration: 998, cost : 0.3754744584166701
# m : 2.448699722825159, b: 1.3800516285429847, iteration: 999, cost : 0.37522060833087606


# learning rate 0.01
# m : 2.0220141121089075, b: 2.920522070246898, iteration: 989, cost : 0.0009087100950040815
# m : 2.021939683230134, b: 2.9207907821154255, iteration: 990, cost : 0.0009025758535948542
# m : 2.021865505992579, b: 2.921058585479309, iteration: 991, cost : 0.0008964830213411816
# m : 2.021791579545453, b: 2.921325483410168, iteration: 992, cost : 0.000890431318710685
# m : 2.021717903040843, b: 2.9215914789692374, iteration: 993, cost : 0.0008844204680584756
# m : 2.0216444756337033, b: 2.921856575207402, iteration: 994, cost : 0.0008784501936131959
# m : 2.0215712964818446, b: 2.922120775165232, iteration: 995, cost : 0.0008725202214656341 
# m : 2.021498364745925, b: 2.922384081873017, iteration: 996, cost : 0.0008666302795551722
# m : 2.0214256795894405, b: 2.922646498350801, iteration: 997, cost : 0.0008607800976581777
# m : 2.0213532401787155, b: 2.9229080276084183, iteration: 998, cost : 0.0008549694073748346
# m : 2.021281045682893, b: 2.923168672645527, iteration: 999, cost : 0.0008491979421174589