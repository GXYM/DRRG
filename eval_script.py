#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import namedtuple
import rrc_evaluation_funcs
import importlib
import math
import numpy as np
import Polygon as plg
from shapely.geometry import Polygon as shapely_poly
import concurrent.futures
from arg_parser import PARAMS
from tqdm import tqdm
import multiprocess as mp
import editdistance
import cv2
def evaluation_imports():
    """
    evaluation_imports: Dictionary ( key = module name , value = alias  )  with python modules used in the evaluation. 
    """    
    return {
            'Polygon':'plg',
            'numpy':'np',
            }

def default_evaluation_params():
    """
    default_evaluation_params: Default parameters to use for the validation and evaluation.
    """
    return {
            'AREA_RECALL_CONSTRAINT' : 0.5,
            'AREA_PRECISION_CONSTRAINT' :0.5,
            'EV_PARAM_IND_CENTER_DIFF_THR': 1,
            'IOU_CONSTRAINT' :0.5,
            'GT_SAMPLE_NAME_2_ID':'.*([0-9]+).*',
            'DET_SAMPLE_NAME_2_ID':'.*([0-9]+).*',
            'GT_LTRB': False, # LTRB: 2points(left,top,right,bottom) or 4 points(x1,y1,x2,y2,x3,y3,x4,y4)
            'GT_CRLF': False, # Lines are delimited by Windows CRLF format
            'DET_LTRB': False, # LTRB: 2points(left,top,right,bottom) or 4 points(x1,y1,x2,y2,x3,y3,x4,y4)
            'DET_CRLF': False, # Lines are delimited by Windows CRLF format
            'CONFIDENCES': False, # Detections must include confidence value. AP will be calculated
            'TRANSCRIPTION': False, # Does prediction has transcription or not
            'PER_SAMPLE_RESULTS': True, # Generate per sample results and produce data for visualization
            }

def validate_data(gtFilePath, submFilePath,evaluationParams):
    """
    Method validate_data: validates that all files in the results folder are correct (have the correct name contents).
                            Validates also that there are no missing files in the folder.
                            If some error detected, the method raises the error
    """
    print(gtFilePath)
    # input()
    gt = rrc_evaluation_funcs.load_zip_file(gtFilePath,evaluationParams['GT_SAMPLE_NAME_2_ID'])

    subm = rrc_evaluation_funcs.load_zip_file(submFilePath,evaluationParams['DET_SAMPLE_NAME_2_ID'],True)
    
    #Validate format of GroundTruth
    for k in gt:
        rrc_evaluation_funcs.validate_lines_in_file(k,gt[k],evaluationParams['GT_CRLF'],evaluationParams['GT_LTRB'],True)

    #Validate format of results
    evaluationParams['TRANSCRIPTION'] = PARAMS.TRANSCRIPTION
    print(evaluationParams['TRANSCRIPTION'])
    for k in subm:
        if (k in gt) == False :
            pass
            # raise Exception("The sample %s not present in GT" %k)
        else:
            rrc_evaluation_funcs.validate_lines_in_file(k,subm[k],evaluationParams['DET_CRLF'],evaluationParams['DET_LTRB'],evaluationParams['TRANSCRIPTION'],evaluationParams['CONFIDENCES'])

        
def evaluate_method(gtFilePath, submFilePath, evaluationParams):
    """
    Method evaluate_method: evaluate method and returns the results
        Results. Dictionary with the following values:
        - method (required)  Global method metrics. Ex: { 'Precision':0.8,'Recall':0.9 }
        - samples (optional) Per sample metrics. Ex: {'sample1' : { 'Precision':0.8,'Recall':0.9 } , 'sample2' : { 'Precision':0.8,'Recall':0.9 }
    """    
    for module,alias in evaluation_imports().items():
        globals()[alias] = importlib.import_module(module)    
    
    def polygon_from_points(points):
        """
        Returns a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
        """        
        resBoxes=np.empty([1,8],dtype='int32')
        resBoxes[0,0]=int(points[0])
        resBoxes[0,4]=int(points[1])
        resBoxes[0,1]=int(points[2])
        resBoxes[0,5]=int(points[3])
        resBoxes[0,2]=int(points[4])
        resBoxes[0,6]=int(points[5])
        resBoxes[0,3]=int(points[6])
        resBoxes[0,7]=int(points[7])
        pointMat = resBoxes[0].reshape([2,4]).T
        return plg.Polygon( pointMat)    
    
    def rectangle_to_polygon(rect):
        resBoxes=np.empty([1,8],dtype='int32')
        resBoxes[0,0]=int(rect.xmin)
        resBoxes[0,4]=int(rect.ymin)
        resBoxes[0,1]=int(rect.xmax)
        resBoxes[0,5]=int(rect.ymin)
        resBoxes[0,2]=int(rect.xmax)
        resBoxes[0,6]=int(rect.ymax)
        resBoxes[0,3]=int(rect.xmin)
        resBoxes[0,7]=int(rect.ymax)
        pointMat = resBoxes[0].reshape([2,4]).T     
        return plg.Polygon( pointMat)
    
    def rectangle_to_points(rect):
        points = [int(rect.xmin), int(rect.ymax), int(rect.xmax), int(rect.ymax), int(rect.xmax), int(rect.ymin), int(rect.xmin), int(rect.ymin)]
        return points

    def polygon_to_points(pol):
        pointMat = []
        for p in pol:
            for i in range(len(p)):
                pointMat.extend(p[i])
        return pointMat
        
    def get_intersection(pD,pG):
        pInt = pD & pG
        if len(pInt) == 0:
            return 0
        return pInt.area()
    
    def compute_ap(confList, matchList,numGtCare):
        correct = 0
        AP = 0
        if len(confList)>0:
            confList = np.array(confList)
            matchList = np.array(matchList)
            sorted_ind = np.argsort(-confList)
            confList = confList[sorted_ind]
            matchList = matchList[sorted_ind]
            for n in range(len(confList)):
                match = matchList[n]
                if match:
                    correct += 1
                    AP += float(correct)/(n + 1)
            if numGtCare>0:
                AP /= numGtCare     
        return AP

    def point_distance(a, b):
        distx = math.fabs(a[0] - b[0])
        disty = math.fabs(a[1] - b[1])
        return math.sqrt(distx * distx + disty * disty)

    def diag(points):
        diag1 = point_distance((points[0], points[1]), (points[4], points[5]))
        diag2 = point_distance((points[2], points[3]), (points[6], points[7]))
        return (diag1 + diag2) / 2

    def center_distance(p1, p2):
        return point_distance(p1.center(), p2.center())
    
    def get_midpoints(p1,p2):
        return ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
    
    def get_angle_3pt(a, b, c):
        """Counterclockwise angle in degrees by turning from a to c around b
            Returns a float between 0.0 and 360.0"""
        ang = math.degrees(
            math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
        return ang + 360 if ang < 0 else ang
    
    def gtBoxtoChars(num, points):
        chars = []     
        assert len(points) == 8
        p1 = get_midpoints([points[0],points[1]], [points[6],points[7]])
        p2 = get_midpoints([points[2],points[3]], [points[4],points[5]])
        unitx = (p2[0] - p1[0]) / num
        unity = (p2[1] - p1[1]) / num
        for i in range(num):
            x = p1[0] + unitx/2 + unitx * i
            y = p1[1] + unity/2 + unity * i
            chars.append((x,y))
        return chars

    def char_fill(detNums, matchMat,detPols,gtCharPoints,gtCharCounts):
        for detNum in detNums:
            detPol = detPols[detNum]
            for gtNum, gtChars in enumerate(gtCharPoints):
                if matchMat[gtNum, detNum] == 1:
                    for gtCharNum, gtChar in enumerate(gtChars):
                        if detPol.polygon.isInside(gtChar[0], gtChar[1]):
                            gtCharCounts[gtNum][detNum][gtCharNum] = 1

    def one_to_one_match(row, col,recallMat,precisionMat):
        cont = 0
        for j in range(len(recallMat[0])):    
            if recallMat[row,j] >= evaluationParams['AREA_RECALL_CONSTRAINT'] and precisionMat[row,j] >= evaluationParams['AREA_PRECISION_CONSTRAINT'] :
                cont = cont +1
        if (cont != 1):
            return False
        cont = 0
        for i in range(len(recallMat)):    
            if recallMat[i,col] >= evaluationParams['AREA_RECALL_CONSTRAINT'] and precisionMat[i,col] >= evaluationParams['AREA_PRECISION_CONSTRAINT'] :
                cont = cont +1
        if (cont != 1):
            return False
        
        if recallMat[row,col] >= evaluationParams['AREA_RECALL_CONSTRAINT'] and precisionMat[row,col] >= evaluationParams['AREA_PRECISION_CONSTRAINT'] :
            return True
        return False

    def one_to_many_match(gtNum,recallMat,precisionMat,detDontCarePolsNum,detPolPoints,gtExcludeMat,detExcludeMat,detTrans):
        many_sum = 0
        detRects = []
        for detNum in range(len(recallMat[0])):
            if detNum not in detDontCarePolsNum and gtExcludeMat[gtNum] == 0 and detExcludeMat[detNum] == 0:
                if precisionMat[gtNum,detNum] >= evaluationParams['AREA_PRECISION_CONSTRAINT']:
                    many_sum += recallMat[gtNum,detNum]
                    detRects.append(detNum)
        if many_sum >= evaluationParams['AREA_RECALL_CONSTRAINT'] and len(detRects) >= 2:
            pivots = []
            for matchDet in detRects:
                ############## Modify by Tran Thuyen 21/05/2021 ##############
                # pD = polygon_from_points(detPolPoints[matchDet])
                pD = MY_POLY(detPolPoints[matchDet],detTrans[matchDet]).make_polygon_obj()
                ##############################################################
                pivots.append([get_midpoints(pD[0][0], pD[0][3]), pD.center()])
            for i in range(len(pivots)):
                for k in range(len(pivots)):
                    if k == i:
                        continue
                    angle = get_angle_3pt(pivots[i][0], pivots[k][1], pivots[i][1])
                    if angle > 180:
                        angle = 360 - angle
                    if min(angle, 180 - angle) >= 45:
                        return False, []
            return True, detRects
        else:
            return False, []

    def many_to_one_match(detNum,recallMat,precisionMat,gtPols,gtDontCarePolsNum,gtExcludeMat,detExcludeMat):
        many_sum = 0
        gtRects = []
        for gtNum in range(len(recallMat)):
            if gtNum not in gtDontCarePolsNum and gtExcludeMat[gtNum] == 0 and detExcludeMat[detNum] == 0:
                if recallMat[gtNum,detNum] >= evaluationParams['AREA_RECALL_CONSTRAINT']:
                    many_sum += precisionMat[gtNum,detNum]
                    gtRects.append(gtNum)
        if many_sum >= evaluationParams['AREA_PRECISION_CONSTRAINT'] and len(gtRects) >= 2:
            pivots = []
            for matchGt in gtRects:
                pG = gtPols[matchGt].polygon
                pivots.append([get_midpoints(pG[0][0], pG[0][3]), pG.center()])
            for i in range(len(pivots)):
                for k in range(len(pivots)):
                    if k == i:
                        continue
                    angle = get_angle_3pt(pivots[i][0], pivots[k][1], pivots[i][1])
                    if angle > 180:
                        angle = 360 - angle
                    if min(angle, 180 - angle) >= 45:
                        return False, []
            return True, gtRects
        else:
            return False, []

    def get_union(pD,pG):
        areaA = pD.area();
        areaB = pG.area();
        return areaA + areaB - get_intersection(pD, pG);

    def get_intersection_over_union(pD,pG):
        try:
            return get_intersection(pD, pG) / get_union(pD, pG);
        except:
            return 0

    # global evalute
    def evalute(resFile,gt,subm,evaluationParams,validating_art=False,constraint_name=None):

        arrGlobalConfidences = []
        arrGlobalMatches = []
        
        gtFile = rrc_evaluation_funcs.decode_utf8(gt[resFile])
        recall = 0
        precision = 0
        hmean = 0    
        recallAccum = 0.
        precisionAccum = 0.
        detCorrect = 0
        detCorrectUpper = 0
        iouMat = np.empty([1,1])

        detMatched = 0
        numGtCare = 0
        numDetCare = 0
        recallMat = np.empty([1,1])
        precisionMat = np.empty([1,1])
        matchMat = np.zeros([1,1])
        
        gtPols = []
        detPols = []
        gtTrans = []
        detTrans = []        
        gtPolPoints = []
        detPolPoints = []  
        
        gtArts = []

        # pseudo character centers
        gtCharPoints = []
        gtCharCounts = []
        
        # visualization
        charCounts = np.zeros([1,1])
        recallScore = list()
        precisionScore = list()

        #Array of Ground Truth Polygons' keys marked as don't Care
        gtDontCarePolsNum = []
        #Array of Detected Polygons' matched with a don't Care GT
        detDontCarePolsNum = []
        

        gtDontCarePolsNum_NED = [] #Array of Ground Truth Polygons' keys marked as don't Care
        detDontCarePolsNum_NED = [] #Array of Detected Polygons' matched with a don't Care GT

        pairs = [] 
        pairs_NED = []
        detMatchedNums = []
        gtExcludeNums = []
        
        arrSampleConfidences = [];
        arrSampleMatch = [];
        sampleAP = 0;

        ned=0;
        nedSum = 0;
        nedElements = 0;
        
        nedUpper=0;
        nedSumUpper = 0;
        nedElementsUpper = 0;

        evaluationLog = ""
        # get_tl_line_values_from_file_contents(content,CRLF=True,LTRB=True,withTranscription=False,withConfidence=False,imWidth=0,imHeight=0,sort_by_confidences=True):
        # pointsList,_,transcriptionsList = rrc_evaluation_funcs.get_tl_line_values_from_file_contents(gtFile, evaluationParams['GT_CRLF'], evaluationParams['GT_LTRB'], True, False)
        pointsList,_,transcriptionsList,artList = rrc_evaluation_funcs.get_tl_line_values_from_dict(gtFile, evaluationParams['GT_CRLF'], evaluationParams['GT_LTRB'], True, False)
        for n in range(len(pointsList)):
            points = pointsList[n]
            transcription = transcriptionsList[n]
            art = artList[n]
            dontCare = transcription == "###"
            if len(transcription) == 0:
                continue
            if evaluationParams['GT_LTRB']:
                gtRect = Rectangle(*points)
                gtPol = rectangle_to_polygon(gtRect)
                points = polygon_to_points(gtPol)
            else:
                ############## Modify by Tran Thuyen 21/05/2021 ##############
                # gtPol = polygon_from_points(points)
                gtPol = MY_POLY(points,transcription,art)
                ##############################################################
            gtPols.append(gtPol)
            if dontCare:
                gtDontCarePolsNum.append( len(gtPols)-1 )
                gtDontCarePolsNum_NED.append( len(gtPols)-1 )
                gtPolPoints.append(points)
                gtCharPoints.append([])
            else:
                gtCharSize = len(transcription)
                aspect_ratio = gtPol.polygon.aspectRatio()
                if aspect_ratio > 1.5:
                    points_ver =  [points[6], points[7], points[0], points[1], points[2], points[3], points[4], points[5]]
                    gtPolPoints.append(points_ver)
                    gtCharPoints.append(gtBoxtoChars(gtCharSize, points_ver))
                else:
                    gtCharPoints.append(gtBoxtoChars(gtCharSize, points))
                    gtPolPoints.append(points)
            gtTrans.append(transcription)
            gtArts.append(art)
        evaluationLog += "GT polygons: " + str(len(gtPols)) + (" (" + str(len(gtDontCarePolsNum)) + " don't care)\n" if len(gtDontCarePolsNum)>0 else "\n")

        # GT Don't Care overlap
        for DontCare in gtDontCarePolsNum:
            for gtNum in list(set(range(len(gtPols))) - set(gtDontCarePolsNum)):
                if get_intersection(gtPols[gtNum].polygon, gtPols[DontCare].polygon) > 0:
                    gtPols[DontCare].polygon -= gtPols[gtNum].polygon

        if resFile in subm:
            
            detFile = rrc_evaluation_funcs.decode_utf8(subm[resFile]) 

            # pointsList,_,transcriptionsList,artList = rrc_evaluation_funcs.get_tl_line_values_from_dict(detFile,evaluationParams['DET_CRLF'],evaluationParams['DET_LTRB'],evaluationParams['TRANSCRIPTION'],evaluationParams['CONFIDENCES'])
            pointsList,confidencesList,transcriptionsList = rrc_evaluation_funcs.get_tl_line_values_from_file_contents(detFile,evaluationParams['DET_CRLF'],evaluationParams['DET_LTRB'],evaluationParams['TRANSCRIPTION'],evaluationParams['CONFIDENCES'])
            for n in range(len(pointsList)):
                points = pointsList[n]
                transcription = transcriptionsList[n]
                if evaluationParams['DET_LTRB']:
                    detRect = Rectangle(*points)
                    detPol = rectangle_to_polygon(detRect)
                    points = polygon_to_points(detPol)
                    detPols.append(detPol)
                    detPolPoints.append(points)
                    detTrans.append(transcription)
                else:
                    ############## Modify by Tran Thuyen 21/05/2021 ##############
                    # detPol = polygon_from_points(points)
                    try:       
                        if len(points) %2 != 0:
                            points = points[-1]
                        points = np.clip(points,0,a_max = None).tolist()
                        detPol = MY_POLY(points,transcription)
                        
                        overlapList = np.array([get_intersection_over_union(detPol.polygon,gtPols[gtNum].polygon) for gtNum in range(len(gtPols))])
                        if len(overlapList) > 0 and validating_art:
                            overlapList = np.array(overlapList)
                            popList = np.where(overlapList > 0.1)[0]
                            argMax = np.argmax(overlapList)
                            if gtPols[argMax].art == False:
                                continue
                        detPols.append(detPol)
                        detPolPoints.append(points)
                        detTrans.append(transcription)
                    except Exception as E:
                        print(E)
                    ##############################################################

            ######## Clear non art ########
            if validating_art:
                gtPols = []
                gtTrans = []
                gtPolPoints = []
                gtCharPoints = []
                gtCharCounts = []
                gtDontCarePolsNum = []
                gtDontCarePolsNum_NED = []
                evaluationLog = ""
                pointsList,_,transcriptionsList,artList = rrc_evaluation_funcs.get_tl_line_values_from_dict(gtFile, evaluationParams['GT_CRLF'], evaluationParams['GT_LTRB'], True, False)
                for n in range(len(pointsList)):
                    points = pointsList[n]
                    transcription = transcriptionsList[n]
                    art = artList[n]
                    dontCare = transcription == "###"
                    if len(transcription) == 0:
                        continue
                    if not art:
                        continue
                    if evaluationParams['GT_LTRB']:
                        gtRect = Rectangle(*points)
                        gtPol = rectangle_to_polygon(gtRect)
                        points = polygon_to_points(gtPol)
                    else:
                        ############## Modify by Tran Thuyen 21/05/2021 ##############
                        # gtPol = polygon_from_points(points)
                        gtPol = MY_POLY(points,transcription,art)
                        ##############################################################
                    gtPols.append(gtPol)
                    if dontCare:
                        gtDontCarePolsNum.append( len(gtPols)-1 )
                        gtDontCarePolsNum_NED.append( len(gtPols)-1 )
                        gtPolPoints.append(points)
                        gtCharPoints.append([])
                    else:
                        gtCharSize = len(transcription)
                        aspect_ratio = gtPol.polygon.aspectRatio()
                        if aspect_ratio > 1.5:
                            points_ver =  [points[6], points[7], points[0], points[1], points[2], points[3], points[4], points[5]]
                            gtPolPoints.append(points_ver)
                            gtCharPoints.append(gtBoxtoChars(gtCharSize, points_ver))
                        else:
                            gtCharPoints.append(gtBoxtoChars(gtCharSize, points))
                            gtPolPoints.append(points)
                    gtTrans.append(transcription)
                    gtArts.append(art)
                evaluationLog += "GT polygons: " + str(len(gtPols)) + (" (" + str(len(gtDontCarePolsNum)) + " don't care)\n" if len(gtDontCarePolsNum)>0 else "\n")

                # GT Don't Care overlap
                for DontCare in gtDontCarePolsNum:
                    for gtNum in list(set(range(len(gtPols))) - set(gtDontCarePolsNum)):
                        if get_intersection(gtPols[gtNum].polygon, gtPols[DontCare].polygon) > 0:
                            gtPols[DontCare].polygon -= gtPols[gtNum].polygon                
            
            ###############################

            evaluationLog += "DET polygons: " + str(len(detPols))

            if len(gtPols)>0 and len(detPols)>0:
                #Calculate IoU and precision matrixs
                outputShape=[len(gtPols),len(detPols)]
                iouMat = np.empty(outputShape)
                recallMat = np.empty(outputShape)
                precisionMat = np.empty(outputShape)
                matchMat = np.zeros(outputShape)
                matchMat_NED = np.zeros(outputShape)
                gtRectMat = np.zeros(len(gtPols),np.int8)
                detRectMat = np.zeros(len(detPols),np.int8)
                gtExcludeMat = np.zeros(len(gtPols),np.int8)
                detExcludeMat = np.zeros(len(detPols),np.int8)
                for gtNum in range(len(gtPols)):
                    detCharCounts = []
                    for detNum in range(len(detPols)):
                        pG = gtPols[gtNum].polygon
                        pD = detPols[detNum].polygon
                        intersected_area = get_intersection(pD,pG)
                        recallMat[gtNum,detNum] = 0 if pG.area()==0 else intersected_area / pG.area()
                        precisionMat[gtNum,detNum] = 0 if pD.area()==0 else intersected_area / pD.area()
                        detCharCounts.append(np.zeros(len(gtCharPoints[gtNum])))
                    gtCharCounts.append(detCharCounts)
                    
                # Find detection Don't Care
                if len(gtDontCarePolsNum)>0 :
                    for detNum in range(len(detPols)):
                        # many-to-one
                        many_sum = 0
                        for gtNum in gtDontCarePolsNum:
                            if recallMat[gtNum, detNum] > evaluationParams['AREA_RECALL_CONSTRAINT']:
                                many_sum += precisionMat[gtNum, detNum]
                        if many_sum >= evaluationParams['AREA_PRECISION_CONSTRAINT']:
                            detDontCarePolsNum.append(detNum)
                        else:
                            for gtNum in gtDontCarePolsNum:
                                if precisionMat[gtNum, detNum] > evaluationParams['AREA_PRECISION_CONSTRAINT']:
                                    detDontCarePolsNum.append(detNum)
                                    break
                        # many-to-one for mixed DC and non-DC
                        for gtNum in gtDontCarePolsNum:
                            if recallMat[gtNum, detNum] > 0:
                                detPols[detNum].polygon -= gtPols[gtNum].polygon
                    for dontCarePol in gtDontCarePolsNum_NED:
                        dontCarePol = gtPols[dontCarePol].polygon
                        intersected_area = get_intersection(dontCarePol,detPol.polygon)
                        pdDimensions = detPol.polygon.area()
                        precision = 0 if pdDimensions == 0 else intersected_area / pdDimensions
                        if (precision > evaluationParams['AREA_PRECISION_CONSTRAINT'] ):
                            detDontCarePolsNum_NED.append( len(detPols)-1 )
                            break                                    
                    evaluationLog += " (" + str(len(detDontCarePolsNum)) + " don't care)\n" if len(detDontCarePolsNum)>0 else "\n"
                
                # Recalculate matrices
                for gtNum in range(len(gtPols)):
                    for detNum in range(len(detPols)):
                        pG = gtPols[gtNum].polygon
                        pD = detPols[detNum].polygon
                        intersected_area = get_intersection(pD,pG)
                        recallMat[gtNum,detNum] = 0 if pG.area()==0 else intersected_area / pG.area()
                        precisionMat[gtNum,detNum] = 0 if pD.area()==0 else intersected_area / pD.area()
                        iouMat[gtNum,detNum] = get_intersection_over_union(pD,pG)
                        ###### NED score by MLT_task_4_1_0.py added 14/10/2021 ######
                        if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 and gtNum not in gtDontCarePolsNum_NED and detNum not in detDontCarePolsNum_NED :
                            if iouMat[gtNum,detNum]>evaluationParams['IOU_CONSTRAINT']:
                                gtRectMat[gtNum] = 1
                                detRectMat[detNum] = 1
                                #detection matched only if transcription is equal (case insensitive)
                                correct_ = gtTrans[gtNum] == detTrans[detNum]
                                correctUpper = gtTrans[gtNum].upper() == detTrans[detNum].upper()
                                
                                detCorrect += (1 if correct_ else 0)
                                detCorrectUpper += (1 if correctUpper else 0)
                                if correctUpper:
                                    detMatchedNums.append(detNum)
                                pairs_NED.append({'gt':[gtNum],'det':[detNum],'type':'','correct':correctUpper})
                                evaluationLog += "Match GT #" + str(gtNum) + " with Det #" + str(detNum) + " trancription correct: " + str(correctUpper) + "\n"

                                #Calculate also 1-NED
                                distance = editdistance.eval(gtTrans[gtNum],detTrans[detNum])
                                nedSum += 0 if len(gtTrans[gtNum])==0 and len(detTrans[detNum])==0 else float(distance) / max( len(gtTrans[gtNum]), len(detTrans[detNum]) )
                                nedElements += 1
                                distanceUpper = editdistance.eval(gtTrans[gtNum].upper(),detTrans[detNum].upper())
                                nedSumUpper += 0 if len(gtTrans[gtNum])==0 and len(detTrans[detNum])==0 else float(distanceUpper) / max( len(gtTrans[gtNum]), len(detTrans[detNum]) )
                                nedElementsUpper += 1

                        ############################################################

                ############ Optimize for loop ############
                for gtNum in range(len(gtPols)):
                    if gtRectMat[gtNum] == 0 and gtNum not in gtDontCarePolsNum_NED :
                        nedSum += 1
                        nedElements += 1
                        nedSumUpper += 1
                        nedElementsUpper += 1

                
                for detNum in range(len(detPols)):
                    if detRectMat[detNum] == 0 and detNum not in detDontCarePolsNum_NED :
                        nedSum += 1
                        nedElements += 1 
                        nedSumUpper += 1
                        nedElementsUpper += 1                            

                # Find many-to-one matches
                evaluationLog += "Find many-to-one matches\n"
                for detNum in range(len(detPols)):
                    if detNum not in detDontCarePolsNum:
                        match, matchesGt = many_to_one_match(detNum,recallMat,precisionMat,gtPols,gtDontCarePolsNum,gtExcludeMat,detExcludeMat)
                        if match:
                            pairs.append({'gt':matchesGt, 'det':[detNum], 'type':'MO'})
                            evaluationLog += "Match GT #" + str(matchesGt) + " with Det #" + str(detNum) + "\n"
                            
                # Find one-to-one matches
                evaluationLog += "Find one-to-one matches\n"
                for gtNum in range(len(gtPols)):
                    for detNum in range(len(detPols)):
                        if gtNum not in gtDontCarePolsNum and detNum not in detDontCarePolsNum :
                            match = one_to_one_match(gtNum, detNum,recallMat,precisionMat)
                            if match:
                                normDist = center_distance(gtPols[gtNum].polygon, detPols[detNum].polygon);
                                normDist /= diag(gtPolPoints[gtNum]) + diag(detPolPoints[detNum]);
                                normDist *= 2.0;
                                if normDist < evaluationParams['EV_PARAM_IND_CENTER_DIFF_THR'] :
                                    pairs.append({'gt':[gtNum],'det':[detNum],'type':'OO'})
                                    evaluationLog += "Match GT #" + str(gtNum) + " with Det #" + str(detNum) + "\n"
                                    
                # Find one-to-many matches
                evaluationLog += "Find one-to-many matches\n"
                for gtNum in range(len(gtPols)):
                    if gtNum not in gtDontCarePolsNum:
                        match, matchesDet = one_to_many_match(gtNum,recallMat,precisionMat,detDontCarePolsNum,detPolPoints,gtExcludeMat,detExcludeMat,detTrans)
                        if match:
                            pairs.append({'gt':[gtNum], 'det':matchesDet, 'type':'OM'})
                            evaluationLog += "Match Gt #" + str(gtNum) + " with Det #" + str(matchesDet) + "\n"

                # Fill match matrix
                for pair in pairs:
                    matchMat[pair['gt'],pair['det']] = 1

                for pair in pairs_NED:
                    matchMat_NED[pair['gt'],pair['det']] = 1                

                # Fill character matrix
                char_fill(np.where(matchMat.sum(axis=0) > 0)[0], matchMat,detPols,gtCharPoints,gtCharCounts)

                # Recall score
                for gtNum in range(len(gtRectMat)):
                    if matchMat.sum(axis=1)[gtNum] > 0:
                        recallAccum += len(np.where(sum(gtCharCounts[gtNum]) == 1)[0]) / len(gtCharPoints[gtNum])
                        if len(np.where(sum(gtCharCounts[gtNum]) == 1)[0]) / len(gtCharPoints[gtNum]) < 1:
                            recallScore.append("<font color=red>" + str(len(np.where(sum(gtCharCounts[gtNum]) == 1)[0])) + "/" + str(len(gtCharPoints[gtNum])) + "</font>")
                        else: recallScore.append(str(len(np.where(sum(gtCharCounts[gtNum]) == 1)[0])) + "/" + str(len(gtCharPoints[gtNum])))
                    else: recallScore.append("")

                # Precision score
                for detNum in range(len(detRectMat)):
                    if matchMat.sum(axis=0)[detNum] > 0:
                        detTotal = 0; detContain = 0
                        for gtNum in range(len(gtRectMat)):
                            if matchMat[gtNum, detNum] > 0:
                                detTotal += len(gtCharCounts[gtNum][detNum])
                                detContain += len(np.where(gtCharCounts[gtNum][detNum] == 1)[0])
                        precisionAccum += detContain / detTotal
                        if detContain / detTotal < 1:
                            precisionScore.append("<font color=red>" + str(detContain) + "/" + str(detTotal) + "</font>")
                        else: precisionScore.append(str(detContain) + "/" + str(detTotal))
                    else:
                        precisionScore.append("")

                # Visualization
                charCounts = np.zeros((len(gtRectMat), len(detRectMat)))
                for gtNum in range(len(gtRectMat)):
                    for detNum in range(len(detRectMat)):
                        charCounts[gtNum][detNum] = sum(gtCharCounts[gtNum][detNum])


            if evaluationParams['CONFIDENCES']:
                for detNum in range(len(detPols)):
                    if detNum not in detDontCarePolsNum :
                        match = detNum in detMatchedNums
                        arrSampleConfidences.append(confidencesList[detNum])
                        arrSampleMatch.append(match)
                        arrGlobalConfidences.append(confidencesList[detNum])
                        arrGlobalMatches.append(match)

        numGtCare = (len(gtPols) - len(gtDontCarePolsNum))
        numDetCare = (len(detPols) - len(detDontCarePolsNum))

        numGtCare_NED = (len(gtPols) - len(gtDontCarePolsNum_NED))
        numDetCare_NED = (len(detPols) - len(detDontCarePolsNum_NED))

        if numGtCare == 0:
            recall = float(1)
            precision = float(0) if numDetCare >0 else float(1)
            sampleAP = precision
        else:
            recall = float(recallAccum) / numGtCare
            precision = float(0) if numDetCare==0 else float(precisionAccum) / numDetCare
            if evaluationParams['CONFIDENCES'] and evaluationParams['PER_SAMPLE_RESULTS']:
                sampleAP = compute_ap(arrSampleConfidences, arrSampleMatch, numGtCare )                    

        if numGtCare_NED == 0:
            ned = float(0) if numDetCare_NED >0 else float(1)
            nedUpper = float(0) if numDetCare_NED >0 else float(1)
            nedSum = numDetCare_NED
            nedSumUpper = numDetCare_NED
            nedElements = numDetCare_NED
            nedElementsUpper = numDetCare_NED
        else:
            if(numDetCare_NED==0):
                nedSum = numGtCare_NED
                nedSumUpper = numGtCare_NED
                nedElements = numGtCare_NED
                nedElementsUpper = numGtCare_NED
                ned =  0
                nedUpper =  0
            else:
                ned =  0 if nedElements==0 else 1 - float(nedSum)/nedElements;                   
                nedUpper =  0 if nedElementsUpper==0 else 1 - float(nedSumUpper)/nedElementsUpper;
        hmean = 0 if (precision + recall)==0 else 2.0 * precision * recall / (precision + recall)                

        evaluationLog += "<b>Recall = " + str(round(recallAccum,2)) + " / " + str(numGtCare) + " = " + str(round(recall,2)) + "\n</b>"
        evaluationLog += "<b>Precision = " + str(round(precisionAccum,2)) + " / " + str(numDetCare) + " = "+ str(round(precision,2)) + "\n</b>"
        

        if evaluationParams['PER_SAMPLE_RESULTS']:
             
            # perSampleMetrics[resFile] = 
            perSampleMetrics_resfile = {
                                            'precision':precision,
                                            'recall':recall,
                                            'hmean':hmean,
                                            'pairs':pairs,
                                            'AP':sampleAP,
                                            'ned':ned,
                                            'nedUpper':nedUpper,
                                            'recallMat':[] if len(detPols)>100 else recallMat.tolist(),
                                            'precisionMat':[] if len(detPols)>100 else precisionMat.tolist(),
                                            'gtPolPoints':gtPolPoints,
                                            'detPolPoints':detPolPoints,
                                            'gtCharPoints':gtCharPoints,
                                            'gtCharCounts':[sum(k).tolist() for k in gtCharCounts],
                                            'charCounts': charCounts.tolist(),
                                            'recallScore': recallScore,
                                            'precisionScore': precisionScore,
                                            'gtTrans':gtTrans,
                                            'detTrans':detTrans,
                                            'art':gtArts,
                                            'gtDontCare':gtDontCarePolsNum,
                                            'detDontCare':detDontCarePolsNum,
                                            'evaluationParams': evaluationParams,
                                            'evaluationLog': evaluationLog
                                        }
            
            temp_dict = {}
            if validating_art:
                # print("Here")
                for key,val in perSampleMetrics_resfile.items():
                    # print(key+" "+constraint_name)
                    temp_dict[key+"_ART_"+constraint_name] = val
                perSampleMetrics_resfile = {}
                perSampleMetrics_resfile = temp_dict
        existART = False
        if True in artList:
            existART = True
        perSampleMetrics_resfile['existART'] = existART
        return recallAccum,precisionAccum,numGtCare,numDetCare,perSampleMetrics_resfile,arrGlobalConfidences,arrGlobalMatches,resFile,\
                detCorrect,detCorrectUpper,numGtCare,numDetCare,nedSum,nedElements,nedSumUpper,nedElementsUpper,existART


    def calculatestar(args):
        def calculate(func, args):
            result = func(*args)
            return result
        return calculate(*args)
    perSampleMetrics = {}
    
    methodRecallSum = 0
    methodPrecisionSum = 0
    
    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    
    gt = rrc_evaluation_funcs.load_zip_file(gtFilePath,evaluationParams['GT_SAMPLE_NAME_2_ID'])
    subm = rrc_evaluation_funcs.load_zip_file(submFilePath,evaluationParams['DET_SAMPLE_NAME_2_ID'],True)

    numGlobalCareGt = 0;
    numGlobalCareDet = 0;
    
    matchedSum = 0
    matchedSumUpper = 0
    globalNedSum = 0
    globalNedElements = 0
    globalNedSumUpper = 0
    globalNedElementsUpper = 0    

    arrGlobalConfidences = [];
    arrGlobalMatches = [];
    ### TODO : Optimize using concurrent

    ### Assign jobs
    TASKS = [(evalute, (resFile,gt,subm,evaluationParams)) for resFile in tqdm(gt)]
    with mp.Pool(processes=PARAMS.NUM_WORKERS) as pool:
        with tqdm(total = len(gt)) as pbar:
            for results in pool.map(calculatestar,TASKS):
                methodRecallSum_perfile,methodPrecisionSum_perfile,numGlobalCareGt_perfile,numGlobalCareDet_perfile,\
                perSampleMetrics_resfile,arrGlobalConfidences_perfile,arrGlobalMatches_perfile,resFile,\
                detCorrect_perfile,detCorrectUpper_perfile,numGtCare_perfile,numDetCare_perfile, \
                nedSum_perfile,nedElements_perfile,nedSumUpper_perfile,nedElementsUpper_perfile, \
                existART = results
                
                ### Regular tedeval return
                methodRecallSum += methodRecallSum_perfile
                methodPrecisionSum += methodPrecisionSum_perfile
                numGlobalCareGt += numGlobalCareGt_perfile
                numGlobalCareDet += numGlobalCareDet_perfile
                
                arrGlobalConfidences.append(arrGlobalConfidences_perfile)
                arrGlobalMatches.append(arrGlobalMatches_perfile)

                ### NED return
                globalNedSum += nedSum_perfile
                globalNedElements += nedElements_perfile
                globalNedSumUpper += nedSumUpper_perfile
                globalNedElementsUpper += nedElementsUpper_perfile
                
                
                perSampleMetrics[resFile] = perSampleMetrics_resfile
                pbar.update(1)
    # Compute MAP and MAR
    AP = 0
    if evaluationParams['CONFIDENCES']:
        AP = compute_ap(arrGlobalConfidences, arrGlobalMatches, numGlobalCareGt)

    methodRecall = 0 if numGlobalCareGt == 0 else methodRecallSum/numGlobalCareGt
    methodPrecision = 0 if numGlobalCareDet == 0 else methodPrecisionSum/numGlobalCareDet
    methodHmean = 0 if methodRecall + methodPrecision==0 else 2* methodRecall * methodPrecision / (methodRecall + methodPrecision)

    methodNed = 0 if globalNedElements==0 else 1 - float(globalNedSum)/globalNedElements;
    methodNedUpper = 0 if globalNedElementsUpper==0 else 1 - float(globalNedSumUpper)/globalNedElementsUpper;
    
    methodMetrics = {'recall':methodRecall, 'precision':methodPrecision, 'hmean':methodHmean, 'AP':AP, 'ned':methodNed, 'nedUpped':methodNedUpper  }
    
    resDict = {'calculated':True,'Message':'','method': methodMetrics,'per_sample': perSampleMetrics}
    
    # # Repeat
    # from config.config import constraints
    # for constraint_name,constraint_val in constraints.items():
    #     # perSampleMetrics = {}
        
    #     methodRecallSum = 0
    #     methodPrecisionSum = 0
                
    #     numGlobalCareGt = 0;
    #     numGlobalCareDet = 0;
        
    #     matchedSum = 0
    #     matchedSumUpper = 0
    #     globalNedSum = 0
    #     globalNedElements = 0
    #     globalNedSumUpper = 0
    #     globalNedElementsUpper = 0    

    #     arrGlobalConfidences = [];
    #     arrGlobalMatches = [];
    #     ### TODO : Optimize using concurrent

    #     evaluationParams['AREA_RECALL_CONSTRAINT'] = constraint_val
    #     evaluationParams['AREA_PRECISION_CONSTRAINT'] = constraint_val
        
    #     ### Assign jobs
    #     TASKS = [(evalute, (resFile,gt,subm,evaluationParams,True,constraint_name)) for resFile in gt]
    #     with mp.Pool(processes=PARAMS.NUM_WORKERS) as pool:
    #         with tqdm(total = len(gt)) as pbar:
    #             for results in pool.map(calculatestar,TASKS):
    #                 methodRecallSum_perfile,methodPrecisionSum_perfile,numGlobalCareGt_perfile,numGlobalCareDet_perfile,\
    #                 perSampleMetrics_resfile,arrGlobalConfidences_perfile,arrGlobalMatches_perfile,resFile,\
    #                 detCorrect_perfile,detCorrectUpper_perfile,numGtCare_perfile,numDetCare_perfile, \
    #                 nedSum_perfile,nedElements_perfile,nedSumUpper_perfile,nedElementsUpper_perfile, \
    #                 existART = results
                    
    #                 if existART:
    #                     ### Regular tedeval return
    #                     methodRecallSum += methodRecallSum_perfile
    #                     methodPrecisionSum += methodPrecisionSum_perfile
    #                     numGlobalCareGt += numGlobalCareGt_perfile
    #                     numGlobalCareDet += numGlobalCareDet_perfile
    #                     arrGlobalConfidences.append(arrGlobalConfidences_perfile)
    #                     arrGlobalMatches.append(arrGlobalMatches_perfile)

    #                     ### NED return
    #                     globalNedSum += nedSum_perfile
    #                     globalNedElements += nedElements_perfile
    #                     globalNedSumUpper += nedSumUpper_perfile
    #                     globalNedElementsUpper += nedElementsUpper_perfile
    #                 perSampleMetrics[resFile].update(perSampleMetrics_resfile) ### update exists sample
    #                 pbar.update(1)
    #     # Compute MAP and MAR
        
    #     AP = 0
    #     if evaluationParams['CONFIDENCES']:
    #         AP = compute_ap(arrGlobalConfidences, arrGlobalMatches, numGlobalCareGt)

    #     methodRecall = 0 if numGlobalCareGt == 0 else methodRecallSum/numGlobalCareGt
    #     methodPrecision = 0 if numGlobalCareDet == 0 else methodPrecisionSum/numGlobalCareDet
    #     methodHmean = 0 if methodRecall + methodPrecision==0 else 2* methodRecall * methodPrecision / (methodRecall + methodPrecision)

    #     methodNed = 0 if globalNedElements==0 else 1 - float(globalNedSum)/globalNedElements;
    #     methodNedUpper = 0 if globalNedElementsUpper==0 else 1 - float(globalNedSumUpper)/globalNedElementsUpper;
        
    #     methodMetricsART = {'recallART '+constraint_name:methodRecall, 'precisionART '+constraint_name:methodPrecision, 'hmeanART '+constraint_name:methodHmean, 
    #                         'APART '+constraint_name:AP, 'nedART '+constraint_name:methodNed, 'nedUppedART '+constraint_name:methodNedUpper  }
    #     # methodMetricsART = {'recallART':methodRecall, 'nedART':methodNed, 'nedUppedART':methodNedUpper  }
        
    #     # resDict = {'calculated':True,'Message':'','methodART': methodMetrics,'per_sampleART': perSampleMetrics}    
    #     resDict['method'].update(methodMetricsART)
    #     import json
    #     # with open("test_{}_.json".format(constraint_name),'w') as f:
    #     #     json.dump(perSampleMetrics,f,indent=4)
    #     resDict['per_sample'].update(perSampleMetrics)
    #     # resDict['per_sampleART'] = perSampleMetrics
    
    return resDict;

class MY_POLY():
    def __init__(self,points,transcription,art=False):
        self.points = points
        self.transcription = transcription
        self.art = art
        self.polygon = self.make_polygon_obj()
    def make_polygon_obj(self):
        
        point_x = self.points[0::2]
        point_y = self.points[1::2]
        # in TotalText dataset, there there are under 4 points annotation for Polygon shape
        # so, we have to deal with it

        # if points are given 3, fill last quad points with left bottom coordinates
        if len(point_x) == len(point_y) == 3:
            point_x.append(point_x[0])
            point_y.append(point_y[2])
            self.points.append(point_x[0])
            self.points.append(point_y[2])
            self.num_points = len(self.points) // 2
        # if points are given 2, copy value 2 times
        elif len(point_x) == len(point_y) == 2:
            point_x *= 2
            point_y *= 2
            self.points.append(point_x[1])
            self.points.append(point_y[0])
            self.points.append(point_x[0])
            self.points.append(point_y[1])
            self.num_points = len(self.points) // 2
        # if points are given 1, copy value 4 times
        elif len(point_x) == len(point_y) == 1:
            point_x *= 4
            point_y *= 4
            for _ in range(3):
                self.points.append(point_x[0])
                self.points.append(point_x[0])
            self.num_points = len(self.points) // 2
        return plg.Polygon(np.stack([point_x, point_y], axis=1))



if __name__=='__main__':
        
    rrc_evaluation_funcs.main_evaluation(None, default_evaluation_params, validate_data, evaluate_method, save_name=None)
