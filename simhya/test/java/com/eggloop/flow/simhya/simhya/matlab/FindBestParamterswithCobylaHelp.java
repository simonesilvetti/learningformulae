package com.eggloop.flow.simhya.simhya.matlab;

import com.eggloop.flow.expr.Context;
import com.eggloop.flow.expr.Variable;
import com.eggloop.flow.mitl.MiTL;
import com.eggloop.flow.mitl.MitlPropertiesList;
import com.eggloop.flow.model.Trajectory;
import com.eggloop.flow.numeric.optimization.ObjectiveFunction;
import com.eggloop.flow.numeric.optimization.cobyla2.Calcfc;
import com.eggloop.flow.numeric.optimization.cobyla2.Cobyla;
import com.eggloop.flow.parsers.MitlFactory;
import com.eggloop.flow.simhya.simhya.matlab.genetic.FitnessFunction;
import com.eggloop.flow.simhya.simhya.matlab.genetic.Formula;
import com.eggloop.flow.simhya.simhya.matlab.genetic.FormulaPopulation;
import com.eggloop.flow.simhya.simhya.matlab.genetic.NewFitness;

import java.util.Arrays;
import java.util.stream.IntStream;

import static org.apache.commons.math3.util.MathArrays.concatenate;

public class FindBestParamterswithCobylaHelp implements ObjectiveFunction

{
    private final String[] variables;
    private final double[][] positiveData;
    private final double[][] negativeData;

    private final Formula formula;
    private final double[] lb;
    private final double[] ub;

    FindBestParamterswithCobylaHelp(String[] variables, double[][] positiveData, double[][] negativeData, Formula formula, double[] lb, double[] ub) {
        this.variables = variables;
        this.positiveData = positiveData;
        this.negativeData = negativeData;
        this.formula = formula;
        this.lb = lb;
        this.ub = ub;
    }

    @Override
    public double getValueAt(double... point) {
        Calcfc calcfc = (n, m, x, con) -> {
            x = denormalize(x, lb, ub);
            //Moltiplicazione matrice
            //con = matrixmultiply(createMatrix(n),x,lb[0],ub[0]);
            con[0] = x[0];
            con[1] = x[1];
            con[2] = 100-x[0];
            con[3] = 100-x[1];
            con[4] = x[1] - x[0] -1;
            if(con[4]<0 || con[0]<0.5){
                return 10;
            }
            double a = -robustness(x, point);
            System.out.println(a);
            return a;
        };

        //double[] x = IntStream.range(0, lb.length).mapToDouble(i -> lb[i] + Math.random() * (ub[i] - lb[i])).toArray();
        double[] x = IntStream.range(0, lb.length).mapToDouble(i -> Math.random()).toArray();
        //x=matrixmultiply(createMatrix(x.length),x);
        double rhobeg = 0.5;
        double rhoend = 1E-1;
        int iprint = 1;
        int maxfun = 3500;
        Cobyla.FindMinimum(calcfc, lb.length, 5, x, rhobeg, rhoend, iprint, maxfun);
        return robustness(x, point);
    }

    private double[] matrixmultiply(double[][] matrix, double[] x, double timelb, double timeub) {
        double[] res = new double[matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            double value =0;
            for (int j = 0; j < x.length; j++) {
                value+=matrix[i][j]*x[j];
            }
            res[i]=value;
        }
        res[0]-=timelb;
        res[res.length-1]+=timeub;
        return res;
    }

    private double[][] createMatrix(int n) {
        double[][] matrix = new double[n+1][n];
        matrix[0]=createDiffArrayStart(n);
        for (int i = 1; i < n; i++) {
            matrix[i]=createDiffArray(i,n);
        }
        matrix[n]=createDiffArrayEnd(n);
        return matrix;
    }
    private double[] createDiffArray(int i,int n){
        double[] array = new double[n];
        array[i-1]=-1;
        array[i]=1;
        return array;
    }
    private double[] createDiffArrayStart(int n){
        double[] array = new double[n];
        array[0]=1;
        return array;
    }
    private double[] createDiffArrayEnd(int n){
        double[] array = new double[n];
        array[n-1]=-1;
        return array;
    }


    public double robustness(double[] times, double[] threshold) {
        double[] vv = concatenate(times, threshold);
        double[] p1u1 = computeAverageRobustness(times, positiveData, variables, formula, vv);
        double[] p2u2 = computeAverageRobustness(times, negativeData, variables, formula, vv);
        FitnessFunction fitness = new NewFitness();
        double compute = fitness.compute(p1u1[0], p2u2[0], 0, p1u1[1], p2u2[1], 0);
        return compute;


    }

    public static double[] denormalize(double[] data, double[] lb, double[] ub) {
        return IntStream.range(0, data.length).mapToDouble(i -> data[i] * (ub[i] - lb[i]) + lb[i]).toArray();
    }


    public static double[] computeAverageRobustness(double[] times, double[][] simulate, String[] variables, Formula formula, double[] vv) {
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
        double mean = Arrays.stream(b).sum() / b.length;
        double variance = Arrays.stream(b).map(x -> (x - mean) * (x - mean)).sum() / b.length;
        return new double[]{mean, variance};

    }

//    private static long numberOfTimePars(String[] parN) {
//        return Arrays.stream(parN).filter(x -> x.contains("Tl") || x.contains("Tu")).count();
//    }
//
//    private static double fitnessBootstrap(FormulaPopulation popgen, Formula formula, int[] data1, int[] data2) {
//        int N = data1.length;
//        double p1 = (double)(Arrays.stream(data1).filter(x -> x == 1).sum() + 1)/(double)(N + 2);
//        double p2 = (double)(Arrays.stream(data2).filter(x -> x == 1).sum() + 1) / (double)(N + 2);
//        double u1 = (double)Arrays.stream(data1).filter(x -> x == -1).sum() / (double)(N);
//        double u2 = (double)Arrays.stream(data2).filter(x -> x == -1).sum() / (double)(N);
//        FitnessFunction fitness = new RegularisedLogOddRatioFitness();
//        return fitness.compute(p1, p2, formula.getFormulaSize(), u1, u2, N);
//    }
//
//
//
//    private double[][] combineTimeAndSpace(double[] times, double[][] spaceTrajectories){
//        double[][] res = new double[spaceTrajectories.length+1][];
//        res[0]=times;
//        System.arraycopy(spaceTrajectories, 0, res, 1, spaceTrajectories.length);
//        return res;
//
//    }
//
//
//    double CNDF(double x)
//    {
//        int neg = (x < 0d) ? 1 : 0;
//        if ( neg == 1)
//            x *= -1d;
//
//        double k = (1d / ( 1d + 0.2316419 * x));
//        double y = (((( 1.330274429 * k - 1.821255978) * k + 1.781477937) *
//                k - 0.356563782) * k + 0.319381530) * k;
//        y = 1.0 - 0.398942280401 * Math.exp(-0.5 * x * x) * y;
//
//        return (1d - neg) * y + neg * (1d - y);
//    }
}
