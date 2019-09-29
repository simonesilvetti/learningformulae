package com.eggloop.flow.simhya.simhya.matlab;

import com.eggloop.flow.numeric.optimization.ObjectiveFunction;
import com.eggloop.flow.simhya.simhya.matlab.genetic.FitnessFunction;
import com.eggloop.flow.simhya.simhya.matlab.genetic.Formula;
import com.eggloop.flow.simhya.simhya.matlab.genetic.FormulaPopulation;
import com.eggloop.flow.simhya.simhya.matlab.genetic.RegularisedLogOddRatioFitness;

import java.util.Arrays;



public class FindBestParamtersFromData implements ObjectiveFunction {
    private final FormulaPopulation popgen;
    private final double[][] positiveData;
    private final double[][] negativeData;

    private final double[] times;
    private final Formula formula;
    private final String[] parsTrshld;
    private final double[] timeBounds;

    FindBestParamtersFromData(FormulaPopulation popgen, double[][] positiveData, double[][] negativeData,double[] times, Formula formula, String[] parsTrshld,double[] timeBounds) {
        this.popgen = popgen;
        this.positiveData = positiveData;
        this.negativeData = negativeData;
        this.times = times;
        this.formula = formula;
        this.parsTrshld = parsTrshld;
        this.timeBounds = timeBounds;
    }

    @Override
    public double getValueAt(double... point) {
        double[] nP = reconvertTimeBounds(parsTrshld, point, timeBounds);
       // System.out.println(Arrays.toString(nP));
        popgen.setParameters(parsTrshld, nP);
        int[] data1 = new int[positiveData.length];
        for (int i = 0; i < positiveData.length; i++) {
            data1[i]=popgen.modelCheck(combineTimeAndSpace(times,new double[][]{positiveData[i]}),formula);
        }
        int[] data2 = new int[negativeData.length];
        for (int i = 0; i < negativeData.length; i++) {
            data2[i]=popgen.modelCheck(combineTimeAndSpace(times,new double[][]{negativeData[i]}),formula);
        }
        double v = fitnessBootstrap(formula, data1, data2);
        //System.out.println(v);
        return v;
    }


    private double[] reconvertTimeBounds(String[] parN, double[] newP, double[] timeBounds) {
        double[] nP = newP;
        long n = numberOfTimePars(parN);
        for (int i = 0; i < n; i += 2) {
            nP[i] = Math.max(timeBounds[0], newP[i] - newP[i + 1]);
            nP[i + 1] = Math.min(timeBounds[1], newP[i] + newP[i + 1]);
        }
        return nP;
    }
    private static long numberOfTimePars(String[] parN) {
        return Arrays.stream(parN).filter(x -> x.contains("Tl") || x.contains("Tu")).count();
    }

    private static double fitnessBootstrap(FormulaPopulation popgen, Formula formula, int[] data1, int[] data2) {
        int N = data1.length;
        double p1 = (double)(Arrays.stream(data1).filter(x -> x == 1).sum() + 1)/(double)(N + 2);
        double p2 = (double)(Arrays.stream(data2).filter(x -> x == 1).sum() + 1) / (double)(N + 2);
        double u1 = (double)Arrays.stream(data1).filter(x -> x == -1).sum() / (double)(N);
        double u2 = (double)Arrays.stream(data2).filter(x -> x == -1).sum() / (double)(N);
        FitnessFunction fitness = new RegularisedLogOddRatioFitness();
        return fitness.compute(p1, p2, formula.getFormulaSize(), u1, u2, N);
    }

    private static double fitnessBootstrap(Formula formula, int[] data1, int[] data2) {
//        int N = data1.length;
//        double p1 = (double)(Arrays.stream(data1).filter(x -> x == 1).count());
//        double p2 = (double)(Arrays.stream(data2).filter(x -> x == 0).count());
//        double value1=p1+p2;
//        p1 = (double)(Arrays.stream(data1).filter(x -> x == 0).count());
//        p2 = (double)(Arrays.stream(data2).filter(x -> x == 1).count());
//        double value2=p1+p2;
//        //double u1 = (double)Arrays.stream(data1).filter(x -> x == -1).sum() / (double)(N);
//        //double u2 = (double)Arrays.stream(data2).filter(x -> x == -1).sum() / (double)(N);
//        //FitnessFunction fitness = new RegularisedLogOddRatioFitness();
//        //return fitness.compute(p1, p2, formula.getFormulaSize(), u1, u2, N);
//        return Math.max(value1,value2);

        int N = data1.length;
        double p1 = (double)(Arrays.stream(data1).filter(x -> x == 1).count() + 1)/(double)(N + 2);
        double p2 = (double)(Arrays.stream(data2).filter(x -> x == 1).count() + 1) / (double)(N + 2);
        double u1 = (double)Arrays.stream(data1).filter(x -> x == -1).count() / (double)(N);
        double u2 = (double)Arrays.stream(data2).filter(x -> x == -1).count() / (double)(N);
        //FitnessFunction fitness = new RegularisedLogOddRatioFitness();
        //double compute = fitness.compute(p1, p2, formula.getFormulaSize(), u1, u2, N);
        //double compute = fitness.compute(Math.max(p1, p2), Math.min(p1, p2), formula.getFormulaSize(), u1, u2, N);
        //double compute = -Math.max(fitness.compute(p1, p2, formula.getFormulaSize(), u1, u2, N),fitness.compute(p2, p1, formula.getFormulaSize(), u2, u1, N));
        //double compute = Math.max(Math.log(p1/p2),Math.log(p2/p1));
        double compute=Math.abs(p1-p2)-1;

        return compute;
    }


    private double[][] combineTimeAndSpace(double[] times, double[][] spaceTrajectories){
        double[][] res = new double[spaceTrajectories.length+1][];
        res[0]=times;
        System.arraycopy(spaceTrajectories, 0, res, 1, spaceTrajectories.length);
        return res;

    }


}
