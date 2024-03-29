package com.eggloop.flow.utils.data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

public class TrajectoryReconstruction {
    private final double[] ds2Times;
    private final double[][] ds2SpatialValues;
    private final double[] ds2Labels;
    private final double validationPercent;
    private final Random ran;
    private double[][] poistiveTrainSet;
    private double[][] poistiveValidationSet;
    private double[][] negativeTrainSet;
    private double[][] negativeValidationSet;

    public TrajectoryReconstruction(double[] ds2Times, double[][] ds2SpatialValues, double[] ds2Labels, double trainPercent, Random ran) {

        this.ds2Times = ds2Times;
        this.ds2SpatialValues = ds2SpatialValues;
        this.ds2Labels = ds2Labels;
        this.validationPercent = trainPercent;
        this.ran = ran;
    }


    public void split() {
        List<Integer> positiveTrajectories = new ArrayList<>();
        List<Integer> negativeTrajectories = new ArrayList<>();
        for (int i = 0; i <ds2Labels.length; i++) {
            if(ds2Labels[i]==-1){
                negativeTrajectories.add(i);
            }else if (ds2Labels[i]==1){
                positiveTrajectories.add(i);
            }
        }
        Collections.shuffle(positiveTrajectories,ran);
        Collections.shuffle(negativeTrajectories,ran);
        int trainPositiveSize= (int)((double) positiveTrajectories.size()*validationPercent);
        int validationPositiveSize= positiveTrajectories.size() - trainPositiveSize;
        int trainNegativeSize= (int)((double) negativeTrajectories.size()*validationPercent);
        int validationNegativeSize= negativeTrajectories.size() - trainNegativeSize;
        poistiveTrainSet = new double[trainPositiveSize][];
        negativeTrainSet = new double[trainNegativeSize][];
        poistiveValidationSet = new double[validationPositiveSize][];
        negativeValidationSet = new double[validationNegativeSize][];
        for (int i = 0; i < trainPositiveSize; i++) {
            poistiveTrainSet[i]=ds2SpatialValues[positiveTrajectories.get(i)];
        }
        for (int i = 0; i < trainNegativeSize; i++) {
            negativeTrainSet[i]=ds2SpatialValues[negativeTrajectories.get(i)];
        }
        for (int i = 0; i < validationPositiveSize; i++) {
            poistiveValidationSet[i]=ds2SpatialValues[positiveTrajectories.get(trainPositiveSize+i)];
        }
        for (int i = 0; i < validationNegativeSize; i++) {
            negativeValidationSet[i]=ds2SpatialValues[negativeTrajectories.get(trainNegativeSize+i)];
        }
    }

    public double[][] getPoistiveTrainSet() {
        return poistiveTrainSet;
    }

    public double[][] getNegativeTrainSet() {
        return negativeTrainSet;
    }

    public double[][] getPoistiveValidationSet() {
        return poistiveValidationSet;
    }

    public double[][] getNegativeValidationSet() {
        return negativeValidationSet;
    }
}
