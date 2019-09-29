package com.eggloop.flow.simhya.simhya.matlab;

import com.eggloop.flow.expr.Context;
import com.eggloop.flow.expr.Variable;
import com.eggloop.flow.mitl.MiTL;
import com.eggloop.flow.mitl.MitlPropertiesList;
import com.eggloop.flow.model.Trajectory;
import com.eggloop.flow.numeric.optimization.ObjectiveFunction;
import com.eggloop.flow.parsers.MitlFactory;
import com.eggloop.flow.simhya.simhya.matlab.genetic.*;

import java.util.Arrays;

public class FindBestParamtersFromAverage  implements ObjectiveFunction {
    private final FormulaPopulation popgen;
    private final String[] variables;
    private final double[][] positiveData;
    private final double[][] negativeData;

    private final double[] times;
    private final Formula formula;
    private final String[] parsTrshld;
    private final double[] timeBounds;

    FindBestParamtersFromAverage(FormulaPopulation popgen, String[] variables, double[][] positiveData, double[][] negativeData,double[] times, Formula formula, String[] parsTrshld,double[] timeBounds) {
        this.popgen = popgen;
        this.variables = variables;
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
        double[] positive = computeAverageRobustness(times, positiveData,variables,formula,  nP);
        double[] negative = computeAverageRobustness(times, negativeData,variables,formula,  nP);
        FitnessFunction fitness = new NewFitness();
        double compute = fitness.compute(positive[0], negative[0], 0, positive[1], negative[1], 0);
        System.out.println(Arrays.toString(nP)+"::::"+compute);
        //System.out.println(compute);
        return compute;

    }


    private double[] reconvertTimeBounds(String[] parN, double[] newP, double[] timeBounds) {
        double[] nP = newP;
        long n = numberOfTimePars(parN);
        for (int i = 0; i < n; i += 2) {
            nP[i] = Math.max(timeBounds[0], newP[i] - newP[i + 1]);
            nP[i + 1] = Math.min(timeBounds[1], newP[i] + newP[i + 1]);
            if(newP[i] - newP[i + 1]==0){
                newP[i + 1]=newP[i]+(timeBounds[1]-timeBounds[0])/100;
            }
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
    public static  double[] computeAverageRobustness(double[] times,double[][] simulate,String[] variables, Formula formula, double[] vv) {
        double[] b = new double[simulate.length];
        Context ns = new Context();
        for (String s : variables) {
            new Variable(s, ns);
        }
        String[] parameters = formula.getParameters();
        StringBuilder builder = new StringBuilder();
        for (int j = 0; j < parameters.length; j++) {
            builder.append("const double ").append(parameters[j]).append("=").append(vv[j]).append(";\n");
        }
        builder.append(formula.toString() + "\n");
        MitlFactory factory = new MitlFactory(ns);
        String text = builder.toString();
        //System.out.println(text);
        MitlPropertiesList l = factory.constructProperties(text);
        MiTL prop = l.getProperties().get(0);

        for (int i = 0; i < simulate.length; i++) {
            Trajectory x = new Trajectory(times, ns, new double[][]{simulate[i]});
            b[i] = prop.evaluateValue(x, 0);
        }
        double mean = Arrays.stream(b).sum()/b.length;
        double variance =  Arrays.stream(b).map(x-> (x-mean)*(x-mean)).sum()/b.length;
        return new double[]{mean,variance};

    }

    double CNDF(double x)
    {
        int neg = (x < 0d) ? 1 : 0;
        if ( neg == 1)
            x *= -1d;

        double k = (1d / ( 1d + 0.2316419 * x));
        double y = (((( 1.330274429 * k - 1.821255978) * k + 1.781477937) *
                k - 0.356563782) * k + 0.319381530) * k;
        y = 1.0 - 0.398942280401 * Math.exp(-0.5 * x * x) * y;

        return (1d - neg) * y + neg * (1d - y);
    }

}
