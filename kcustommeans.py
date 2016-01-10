## k means 

# with custom distance function 
# current : haversine (fast version)

# with weights, and weight limit
# efficiency/accuracy not evaluated
# weighted distance may not be correct for non-euclidean distances

## NEEDS CLEANING UP


import pandas as pd
import numpy as np
from nphaversine import nphaversine
import time



def closest_centroid_haversine(points, centroids, full=False):    
    # CHANGE TO CORRECT HAVERSINE DISTANCE    
    # IF SLOW USE BACK NORMAL DISTANCE?
    
    # try from scipy.spatial.distance import pdist, cdist
    #   or from sklearn.metrics.pairwise import pairwise_distances
#    start = time.clock()
#    from sklearn.metrics.pairwise import pairwise_distances
#    distances = pairwise_distances(points,centroids,nphaversine)
    
    if len(points.shape) == 1:
        points = np.array([points])
    if len(centroids.shape) == 1:
        centroids = np.array([centroids])
        
    
#    start = time.clock()
    ### can try just nphaversine(points[],centroids) loop but maybe no diff
    
    # faster? -> haversine projection trick
    point1 = np.radians(points[:, np.newaxis])
    point2 = np.radians(centroids)
###    d = np.power(np.sin((point2[:,0] - point1[:,0]) / 2) , 2) + np.cos(point1[:,0]) * np.cos(point2[:,0]) * np.power(np.sin((point2[:,1] - point1[:,1]) / 2), 2)
     
    inter = np.cos(np.radians(points[:,0]))[:, np.newaxis] * np.cos(np.radians(centroids[:,0]))   # THIS WAS WRONG FUUUUUUUUUCK

    # CANT BE UNFUCKED
#    d = np.dot(  np.power(np.sin((point2 - point1) /2),2)  ,np.vstack((np.ones(inter.shape),inter)).transpose() )
    
    d = np.power(np.sin((point2- point1) /2),2) 
    d[:,:,1] = d[:,:,1] * inter #inter2.transpose()
    d = np.sum(d,axis=2)
    
    distances = 2 * 6371 * np.arcsin(np.sqrt(d))  
#    time.clock() - start 
    ### TEST AGAINST PAIRWISE  ## SUCCESS 100 TIMES FASTER
    if full == True:
        return np.squeeze(np.argsort(distances,axis=1))
    else:
        return np.argmin(distances, axis=1)    
    

    
def move_centroids(points, closest, centroids,weights):
    """returns the new centroids assigned from the points closest to them"""
    
    #  CHANGE TO WEIGHTED DISTANCE, DONE NEED TEST
#    return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])
    return np.array([np.average(points[closest==k],axis=0,weights = weights[closest==k]) for k in range(centroids.shape[0])])

def add_new_centroids(points,centroids, weights, weightlimit):
    centroidweights = calculate_weights_of_centroids(points,centroids,weights)
#    print centroidweights
    toofat = centroids[np.where(centroidweights > weightlimit)[0]]
    toofat2 = toofat +0.001  # offset by ~2 kilometers
    toofat = toofat - 0.001 
    centroids[np.where(centroidweights > weightlimit)[0]] = toofat
    centroids = np.vstack((centroids,toofat2))
    
    return centroids


def calculate_weights_of_centroids(points,centroids,weights):
    indices = closest_centroid_haversine(points,centroids)
    
    centroidweights = np.empty((centroids.shape[0],1))
    for i in range(len(centroids)):    
        
        centroidweights[i] = weights[indices == i].sum()
        #haversine(points[indices == i],centroids[i])
    return centroidweights


def iteratefinishcriteria(noofiters,topiter):
    if topiter >0:
        noofiters -= topiter
    if noofiters >0:
        return True
    else:
        return False
    
def pickstartingcentroids(points,initnoofcentroids):
    
    temp = np.random.choice(points.shape[0],initnoofcentroids,replace=False)    
    
    return points[temp]


def kcustommeans(points,weights,initnoofcentroids,weightlimit, savefile='NA'):
    firstrun= 1
    if savefile == 'NA':
        centroids = pickstartingcentroids(points,initnoofcentroids)    
        topiter = 8
    else:
        centroids = np.load(savefile+'.npy')
    
    ## form centroids
    oldtemp = np.ones((2,1)) * 3000
    
    while 1:
        ## 
        
        noofiters = 0    
        start = time.clock()
        while 1:
            print 'iter'
            f = open('progress.txt','a')
            f.write('iter- number of centroids ') # python will convert \n to os.linesep
            f.write(str(centroids.shape[0]))
            f.write('\n')
            f.close()
            noofiters +=1
            
            
            
            closest = closest_centroid_haversine(points,centroids)
            
            # check whether centroid has no closest point
            deadcentroids = set(range(centroids.shape[0])).difference(set(closest))
            
            if len(deadcentroids) > 0:
                centroids = np.delete(centroids,list(deadcentroids),axis=0)
                closest = closest_centroid_haversine(points,centroids)     # stopgap measure
#                while len(set(closest_points)) < sorted(list(set(closest_points)))[-1]:
#                    for idx,i in enumerate(sorted(list(set(closest_points)))):
#                        if idx != i:
#                            closest_points[closest_points > idx] = closest_points[closest_points > idx] - (i-idx)
#                            break
            
            centroids = move_centroids(points, closest, centroids,weights)
            
            
            if iteratefinishcriteria(noofiters,topiter):
                print str(noofiters)+' iter(s) taking ' , time.clock() - start , ' seconds'
                qq = str(noofiters)+' iter(s) taking ' , time.clock() - start , ' seconds'
                f = open('progress.txt','a')
                f.write(str(qq)+'\n') # python will convert \n to os.linesep
                f.close()
                break
        topiter -=2
        print topiter
        
        centroidweights = calculate_weights_of_centroids(points,centroids,weights)
        print 'OVERWEIGHT NUMBER: ' + str(np.where(centroidweights>weightlimit)[0].shape[0])
        f = open('progress.txt','a')
        f.write('OVERWEIGHT NUMBER: ' + str(np.where(centroidweights>weightlimit)[0].shape[0]) +'\n')
#        f.write('old: ' + str(np.where(oldtemp>weightlimit)[0].shape[0]) + '  . new: ' + str(np.where(centroidweights>weightlimit)[0].shape[0]) + '\n')
        f.close()
        
        if firstrun == 1:
            firstrun = 0
        elif firstrun == 0:
            # BECAUSE NO OF OVERWEIGHT WILL INCREASE INITIALLY
            if ((np.where(oldtemp>weightlimit)[0].shape[0] - np.where(centroidweights>weightlimit)[0].shape[0]  )*1.0 / np.where(oldtemp>weightlimit)[0].shape[0]) > 0:
                firstrun = -1
        elif  ((np.where(oldtemp>weightlimit)[0].shape[0] - np.where(centroidweights>weightlimit)[0].shape[0]  )*1.0 / np.where(oldtemp>weightlimit)[0].shape[0]) <  0.4:
            print 'no iteration improvement'
            f = open('progress.txt','a')
            f.write('no iteration improvement\n')
            f.write('old: ' + str(np.where(oldtemp>weightlimit)[0].shape[0]) + '  . new: ' + str(np.where(centroidweights>weightlimit)[0].shape[0]) + '\n')
            f.close()
            # do the k means for each overweight
            break # no iteration improvement
        
        
        oldtemp = centroidweights.copy()
        if (np.where(centroidweights>2000)[0].shape[0]  ) == 0   : # hard code bullshit to get some results
            f = open('progress.txt','a')
            f.write('all < 2000\n')
            f.close()
            break
        else:
            print centroidweights[centroidweights>weightlimit]
            f = open('progress.txt','a')
            f.write('1\n')
            f.write(str(centroidweights[centroidweights>weightlimit])+'\n') # python will convert \n to os.linesep
            f.close()

            np.save('centroidprogress' , centroids)

            centroids= add_new_centroids(points,centroids, weights,weightlimit)
            
            
            f = open('progress.txt','a')
            centroidweights = calculate_weights_of_centroids(points,centroids,weights)
            f.write('2\n')
            f.write(str(centroidweights[centroidweights>weightlimit])+'\n') # python will convert \n to os.linesep
            f.close()
            # apparently this is still fucked. adding of new centroids is shit. try a different way
            # if all below 2000, (Examine these points by plotting it and the centroid)
            # k means those points with 2 centroids?
            #
            # try iterate till (oldtemp.shape[0] - temp.shape[0]  )/ oldtemp.shape[0] <  0.4 #40% decrease at least
            # then k means with centroidsweights/1000.0 centroids


    closest = closest_centroid_haversine(points,centroids)        # see above    
    
    np.save('closest' , closest)
    np.save('centroidweights' , centroidweights)
    np.save('centroids' , centroids)
    
#    closest = np.load('closest.npy')
#    centroidweights = np.load('centroidweights.npy')
#    centroids = np.load('centroids.npy')
    
    
    if (np.where(centroidweights>1000)[0].shape[0]  ) > 0   : #
    
    
#        closest = closest_centroid_haversine(points,centroids) 
        f = open('progress.txt','a')
        f.write('Got above 1000, proceeding to cut down those bastards\n')
        f.close()
        for i in np.where(centroidweights>1000)[0]:
            # iterate through bastards and k means them
            numberofnewcentroids = int(np.ceil(centroidweights[i] / weightlimit))
            subpoints = points[closest==i]
            subweights = weights[closest==i]
            
            subcentroids = pickstartingcentroids(subpoints,numberofnewcentroids)
            topiter = 3
            noofiters= 0
            explode = 0
            explode2 = 0
#            if  i == 487:
#                raise 'awd'   
            while 1:
                noofiters +=1
                subclosest = closest_centroid_haversine(subpoints,subcentroids)
                deadcentroids = set(range(subcentroids.shape[0])).difference(set(subclosest))
                if len(deadcentroids) > 0:
                    explode2 +=1
                    noofiters = 0
                    print 'Fucking dead centroid for '+ str(i)
                    print 'points: ' + str(subpoints.shape[0])   + ' numberofnewcentroids: ' + str(numberofnewcentroids)
                                    
                    subcentroids = pickstartingcentroids(subpoints,numberofnewcentroids)
#                    time.sleep(1)
                    if explode2 > 5:
                        numberofnewcentroids +=1
                        subcentroids = pickstartingcentroids(subpoints,numberofnewcentroids)
                        recalcweights = calculate_weights_of_centroids(subpoints,subcentroids,subweights)
                        if (np.where(recalcweights>1000)[0].shape[0]) == 0:
                            break
                    continue
#           
                
                subcentroids = move_centroids(subpoints, subclosest, subcentroids,subweights)
                if iteratefinishcriteria(noofiters,topiter):
                    recalcweights = calculate_weights_of_centroids(subpoints,subcentroids,subweights)
                    if (np.where(recalcweights>1000)[0].shape[0]) == 0:
                        break
                    if explode >5:
                        
                        print 'FUCK '+ str(i)
                        f = open('progress.txt','a')
                        f.write('FUCK '+ str(i) + 'GET EVEN MORE CENTROIDDS' + '\n')
                        f.write('Weights: ' + str(recalcweights) + '\n')
                        f.close()
                        numberofnewcentroids +=1
                        subcentroids = pickstartingcentroids(subpoints,numberofnewcentroids)
                        noofiters = 0
                        
                    if noofiters > 18:
                        subcentroids = pickstartingcentroids(subpoints,numberofnewcentroids)
                        noofiters = 0
                        explode +=1
                        print 'Failed to pick new balanced centroids for '+ str(i)
                        f = open('progress.txt','a')
                        f.write('Failed to pick new balanced centroids for '+ str(i) + '\n')
                        f.write('Weights: ' + str(recalcweights) + '\n')
                        f.close()
            # assign new centroids
            # BUT WAIT ITS FUCKED DONT RECALC CLSOEST YOU'LL FUCK IT UP JUST ASSIGN CLOSEST
            subclosest = closest_centroid_haversine(subpoints,subcentroids)
            
            
            subclosest[subclosest==0] = i
            for j in range(1,numberofnewcentroids):
                subclosest[subclosest==j] = centroids.shape[0]+j
            closest[closest==i] = subclosest
            
            centroids[i,:] = subcentroids[0,:]
            centroids = np.vstack((centroids,subcentroids[1:,:]))
            
                        
    np.save('correctedclosest' , closest)   
                    
    
    
    ## merge centroids
    
    # this recalcs centroid weights wtf
    # need a way to recalc weight using closest
    closest = mergekcustommeanscentroids(closest, points,weights,weightlimit,centroids)


    return  closest
        

def calculate_weights_of_centroids_backwards(closest,centroids,weights):
    centroidweights = np.zeros((centroids.shape[0],1))    
    for i in range(centroids.shape[0]):
        temp = np.where(closest == i)[0]
        centroidweights[i] = weights[temp].sum()
    
    
    return np.squeeze(centroidweights)
    
'''
closest = np.load('closest.npy')
centroidweights = np.load('last/centroidweights.npy')
centroids = np.load('last/centroids.npy')
closest = np.load('last/correctedclosest.npy')
closest_points = closest
'''
            
def mergekcustommeanscentroids(closest_points, points,weights,weightlimit,centroids,depth=3,breadth=1):

        
    # iterate from least filled centroid?

    cutdowntarget =  centroids.shape[0] - 1500 # 1500 is if whole world
    target = 0
    
    centroids = np.ma.array(centroids, mask=False) 
    
#    centroidweights = calculate_weights_of_centroids(points,centroids,weights)[:,0]  # slow
    centroidweights = calculate_weights_of_centroids_backwards(closest_points,centroids,weights) # faster method
    
    centroidweights = np.ma.array(centroidweights, mask=False) 
        
    for i in range(cutdowntarget):
        if target >= len(np.where(centroidweights.mask==False)[0]):
            print 'all weights masked'
            f = open('progress.txt','a')
            f.write('all weights masked \n')
            f.close()
            break
        
#        centroids, flag = mergecentroid(points,weights,centroids,centroidweights,weightlimit,targetted)
        
        
        
        
        for j in range(breadth):
            
            targetted = np.argsort(centroidweights)[target]
            
            pluscounter = 0
            targettedpooled = [targetted]
            centroids[targetted,:] = np.ma.masked  
#            pooledweight = 0
            success = 0
            for k in range(depth):
                if k == 0:
                    pluscounter += j
                    
                # np.vstack((centroids[:targetted,:],centroids[targetted+1:,:])) USE MASKED ARRAY FFS
                # 
                    # masked only for centroids?
                closest = closest_centroid_haversine(np.ma.getdata(centroids[targetted,:]),centroids,full=True) #np.vstack((centroids[:targetted,:],centroids[targetted+1:,:]))
#                pooledweight += centroidweight[targetted]
                
                targettedpooled.append(closest[0])
                
                if (centroidweights[closest[pluscounter]] + centroidweights[targettedpooled[:-1]].sum() +1 - (weightlimit * (len(targettedpooled) -2)) ) < weightlimit:
                    # reassign closest that = targettedcentroid to next centroid
                    # update weight of next centroid (centroidweights)
                    
                    closest_points[closest_points==targettedpooled[0]] = targettedpooled[1]
                    centroidweights[targettedpooled[1]] += centroidweights[targettedpooled[0]]
                    centroidweights[targettedpooled[0]] = 0 # just to be sure
                    
                    # combine current with next,
                    l=0
                    for l in range(1,len(targettedpooled)-1):               #len(targettedpooled)-1,-1,-1):
                         #reassign next to next next by nearest points to next centroid until next < weightlimit
                        
                        #we know current is over weight limit
                        # so we now want to transfer enough to next one
                        pointlistofnext= np.where(closest_points==targettedpooled[l])[0]
                        closesttemp = closest_centroid_haversine(np.ma.getdata(centroids[targettedpooled[l+1],:]),points[closest_points==targettedpooled[l],:],full=True)
                        countertemp = -1            
                        while centroidweights[targettedpooled[l]] > weightlimit:
                            countertemp +=1
                            
                            #if l == len(targettedpooled)-2: # last iter, dont want overload last one
                            #    if ( centroidweights[targettedpooled[l+1]] + weights[pointlistofnext[closesttemp[countertemp]]] )> weightlimit:
                            #        continue  # pray that got at least one point that satisfy both weight limits
                            #else:
                            #    # ensure that the next few points will add up to weightlimit*number, otherwise it's dooming them to failure
                            #    # longer depth even more likely to have problem                                
                            if ( centroidweights[targettedpooled[l+1:]].sum() + weights[pointlistofnext[closesttemp[countertemp]]]  + 1)> (weightlimit * len(targettedpooled[l+1:])):  # + 1 because no points below 1, so must make sure 1 gap
                                continue  # this should work for all l
                            
                            
                            # assign closest to next   to next
                            closest_points[pointlistofnext[closesttemp[countertemp]]] = targettedpooled[l+1]
                            # reassign weights properly
                            centroidweights[targettedpooled[l]] -= weights[pointlistofnext[closesttemp[countertemp]]]
                            centroidweights[targettedpooled[l+1]]   += weights[pointlistofnext[closesttemp[countertemp]]]
                    
                    
                    
                    
                            
                    if centroidweights[targettedpooled[l+1]] > weightlimit:
                        print 'FUCK FUUUCK LAST ONE IS OVERWEIGHT'
                        closesttemp2 = closest_centroid_haversine(np.ma.getdata(centroids[targettedpooled[l],:]),centroids,full=True)
                        
                        for temptemp in range(len(closesttemp2)):
                            if centroidweights[targettedpooled[l]] + centroidweights[closesttemp2[temptemp]] < 2* weightlimit:
                                break
#                        countertemp = -1
                        while centroidweights[targettedpooled[l]] > weightlimit:    
#                            countertemp +=1
                            
                           
                            if  centroidweights[targettedpooled[l]] + weights[closesttemp2[temptemp]]> weightlimit:
                                continue  # pray that got at least one point that satisfy both weight limits
                            
                            
                            # assign closest to next   to next
                            closest_points[closesttemp2[countertemp]] = targettedpooled[l]
                            # reassign weights properly
                            centroidweights[targettedpooled[l]] -= weights[closesttemp2[temptemp]]
                            centroidweights[closesttemp2[temptemp]]   += weights[closesttemp2[temptemp]]
                        
                        
                        
                    # check last one < weightlimit. if not, fuck. (throw to closest one?)
                    
                    success = 1
                    break   # for depth
                else: # move on to  target's target because target not thin enough
                    
                    targetted = closest[pluscounter]
                    
                    centroids[targetted,:] = np.ma.masked  
            
            centroids.mask[targettedpooled[1:],:] = False  # unmask the ahead ones
            if success == 1: 
                print '1 absorbed'
                centroidweights[targettedpooled[0]] = np.ma.masked 
                break # for breadth
                
        
        if success == 0: # can't find, skip target
            centroids.mask[targettedpooled,:] = False # unmask every tried one
            target +=1 # this assumes the larger ones wont go hit the smaller one TODO think of a way, always refresh, detect change to it?
            
        
        # TODO UNMASK CENTROID
        # WHAT TO DO ABOUT DEAD CENTROID -> KEEP  MASKED
        
        
        
        # squeeze closest
        # HOLY SHIT THIS ACTUALLY WORKS, insert nparray
    while len(set(closest_points)) < sorted(list(set(closest_points)))[-1]:
        for idx,i in enumerate(sorted(list(set(closest_points)))):
            if idx != i:
                closest_points[closest_points > idx] = closest_points[closest_points > idx] - (i-idx)
                break

    return closest_points           








weightlimit = 1000.0
origin = np.array([90,0])


dataset['TripId']=0

points = np.vstack((dataset['Latitude'].values,dataset['Longitude'].values)).transpose()
weights = dataset['Weight'].values

closest = kcustommeans(points,weights,int(dataset['Weight'].values.sum()/1000/4),weightlimit)#                     points,weights,initnoofcentroids,weightlimit)

np.save('finalresult' , closest)


centroidweights = calculate_weights_of_centroids_backwards(closest,centroids,weights) 
np.where(centroidweights>weightlimit)

def greedytravelling(points, startingpoint = 0):
    points = np.ma.array(points, mask=False)  
    finalgrouping = np.zeros((points.shape[0],1))
    chosenpoint = startingpoint
    counter = 0
    for i in range(points.shape[0]-1):
        counter += 1
        chosenpointcoords = points[chosenpoint].data
        points[chosenpoint] = np.ma.masked  
        closeindices = np.argsort(nphaversine(chosenpointcoords,points))[0]
        finalgrouping[counter] =  closeindices
        chosenpoint = closeindices
    return finalgrouping





finalgrouping = closest

tripid = np.squeeze(np.zeros(finalgrouping.shape))
giftid = np.squeeze(np.zeros(finalgrouping.shape))

counter = 0
tripcounter = 0
for i in range(int(finalgrouping.max()+1)):
    
    if i%10 == 0:
        print 'Group: ' + str(i)    
    
    members  = np.where(finalgrouping==i)
#    grpmemberdist = haversine(origin,points[members[0]])  # this is sorting by who nearest pole
#    order = np.argsort(grpmemberdist) # this is sorting by who nearest pole
    
    finalgroupingrearrange = np.squeeze(greedytravelling(points[members[0],:])).astype(int)
    
    # DO A MULTIPLE START GREEDY TRAVEL EVAL THE DIST?
    # OR JUST START WITH THE ONE NEAREST THE POLE?
    
    
    
    a = int(finalgroupingrearrange[0])
    z = int(finalgroupingrearrange[-1])
    
    if nphaversine(points[members[0]][a],origin)>nphaversine(points[members[0]][z],origin):
        finalgroupingrearrange = finalgroupingrearrange[::-1]
        
    
    
    tripid[counter:counter+len(members[0])] = tripcounter
    giftid[counter:counter+len(members[0])] = members[0][finalgroupingrearrange]
    counter += len(members[0])
    tripcounter += 1



giftid = giftid+1  # NO 0 GIFTID, DONT RUN THIS TWICE, CHECK WITH giftid.min()
output = pd.DataFrame( data={"GiftId":giftid.astype(int), "TripId":tripid.astype(int)} )

# Use pandas to write the comma-separated output file
output.to_csv( "NEWsantasubmissionkmeans.csv", index=False, quoting=3 )








from haversine import haversine

north_pole = (90,0)
weightlimit = 1000.0

def bb_sort(ll): 
    ll = [[0,north_pole,10]] + ll[:] + [[0,north_pole,10]] 
    for i in range(1,len(ll) - 2):
        lcopy = ll[:]
        lcopy[i], lcopy[i+1] = ll[i+1][:], ll[i][:]                             # swap neighbours
        if path_opt_test(ll[1:-1]) > path_opt_test(lcopy[1:-1]):                # if swap neighbour decrease total distance, do it
            ll = lcopy[:]
    return ll[1:-1]

def path_opt_test(llo):
    f_ = 0.0
    d_ = 0.0
    l_ = north_pole
    for i in range(len(llo)):
        d_ += haversine(l_, llo[i][1])
        f_ += d_ * llo[i][2]
        l_ = llo[i][1]
    d_ += haversine(l_, north_pole)
    f_ += d_ * 10 #sleigh weight for whole trip
    return f_
    

dataset['TripId'] = closest


t_ = closest.max()
trips=dataset[dataset['TripId']==0]
trips=trips.sort_values(['i','j','Longitude','Latitude'])

ou_ = open("submission_opt" + " " + ".csv","w")
ou_.write("TripId,GiftId\n")
bm = 0.0

for s_ in range(1,t_+1):  # for each trip
    trip=dataset[dataset['TripId']==s_]
    trip=trip.sort_values(['Latitude','Longitude'],ascending=[0,1])         # grab the trip members and sort to lat long
    
    a = []
    for x_ in range(len(trip.GiftId)):
       
        a.append([trip.iloc[x_,0],(trip.iloc[x_,1],trip.iloc[x_,2]),trip.iloc[x_,3]])  # generate trip according to lat lon sort
    b = bb_sort(a)                                                              #bb sort trip
    if path_opt_test(a) <= path_opt_test(b):                                    # see which higher
        print("TripId",s_, "No Change", path_opt_test(a) , path_opt_test(b))
        bm += path_opt_test(a)
        for y_ in range(len(a)):
            ou_.write(str(int(s_))+","+str(int(a[y_][0]))+"\n")             # doesnt matter trip starts from 1 or 0
    else:
        print("TripId ", s_, "Optimized", path_opt_test(a) - path_opt_test(b))
        bm += path_opt_test(b)
        for y_ in range(len(b)):
            ou_.write(str(int(s_))+","+str(int(b[y_][0]))+"\n")
ou_.close()







