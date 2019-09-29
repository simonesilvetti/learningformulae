package com.eggloop.flow.simhya.simhya.matlab;

import com.eggloop.flow.numeric.optimization.ObjectiveFunction;
import com.eggloop.flow.simhya.simhya.matlab.genetic.*;

import java.util.Arrays;

public class FindBestParamtersFromSimulations2  implements ObjectiveFunction {
    private final FormulaPopulation popgen;
    private final BreathSimulator simulator1;
    private final BreathSimulator simulator2;
    private final double tf;
    private final Formula formula;
    private final String[] pars;
    private final int samples;
    private final double[] timeBounds;

    public FindBestParamtersFromSimulations2(FormulaPopulation popgen, BreathSimulator simulator1, BreathSimulator simulator2, double tf, Formula formula, String[] pars, int samples, double[] timeBounds) {

        this.popgen = popgen;
        this.simulator1 = simulator1;
        this.simulator2 = simulator2;
        this.tf = tf;
        this.formula = formula;
        this.pars = pars;
        this.samples = samples;
        this.timeBounds = timeBounds;
    }

    @Override
    public double getValueAt(double... point) {
        return OptimisationFunction(popgen,simulator1,simulator2, tf, formula, pars,point, samples,timeBounds);
    }

    private double OptimisationFunction(FormulaPopulation popgen, BreathSimulator simfunc1, BreathSimulator simfunc2, double Tf, Formula formula, String[] parN, double[] newP, int samples, double[] timeBounds) {
        System.out.println("newP: "+Arrays.toString(newP));
        double[] nP = reconvertTimeBounds(parN, newP, timeBounds);
        System.out.println("np: "+Arrays.toString(nP));
        popgen.setParameters(parN, nP);
        int[] data1 = popgen.modelCheck(simfunc1, formula, samples, Tf);
        int[] data2 = popgen.modelCheck(simfunc2, formula, samples, Tf);
        double v = fitnessBootstrap(formula, data1, data2);
        System.out.println(v);
        //System.out.println("nP: "+Arrays.toString(nP));
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
        FitnessFunction fitness = new RegularisedLogOddRatioFitness();
        //double compute = fitness.compute(p1, p2, formula.getFormulaSize(), u1, u2, N);
        //double compute = fitness.compute(Math.max(p1, p2), Math.min(p1, p2), formula.getFormulaSize(), u1, u2, N);
        //double compute = -Math.max(fitness.compute(p1, p2, formula.getFormulaSize(), u1, u2, N),fitness.compute(p2, p1, formula.getFormulaSize(), u2, u1, N));
        double compute = Math.max(Math.log(p1/p2),Math.log(p2/p1));

        return compute;
    }
}
