package com.eggloop.flow.simhya.simhya.matlab;

import com.eggloop.flow.numeric.optimization.ObjectiveFunction;
import com.eggloop.flow.simhya.simhya.matlab.genetic.FitnessFunction;
import com.eggloop.flow.simhya.simhya.matlab.genetic.Formula;
import com.eggloop.flow.simhya.simhya.matlab.genetic.FormulaPopulation;
import com.eggloop.flow.simhya.simhya.matlab.genetic.RegularisedLogOddRatioFitness;

import java.util.Arrays;

public class FindBestParamtersFromSimulations implements ObjectiveFunction {
    private final FormulaPopulation popgen;
    private final BasicSimulator simulator1;
    private final BasicSimulator simulator2;
    private final double tf;
    private final Formula formula;
    private final String[] pars;
    private final int samples;
    private final double[] timeBounds;

    public FindBestParamtersFromSimulations(FormulaPopulation popgen, BasicSimulator simulator1, BasicSimulator simulator2, double tf, Formula formula, String[] pars, int samples, double[] timeBounds) {

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

    private double OptimisationFunction(FormulaPopulation popgen, BasicSimulator simfunc1, BasicSimulator simfunc2, double Tf, Formula formula, String[] parN, double[] newP, int samples, double[] timeBounds) {
        double[] nP = reconvertTimeBounds(parN, newP, timeBounds);
        System.out.println(Arrays.toString(nP));
        popgen.setParameters(parN, nP);
        int[] data1 = popgen.modelCheck(simfunc1, formula, samples, Tf);
        int[] data2 = popgen.modelCheck(simfunc2, formula, samples, Tf);
        double v = fitnessBootstrap(popgen, formula, data1, data2);
        System.out.println(v);
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
}
