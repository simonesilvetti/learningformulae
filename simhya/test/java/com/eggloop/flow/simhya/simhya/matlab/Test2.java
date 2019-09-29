package com.eggloop.flow.simhya.simhya.matlab;

import com.eggloop.flow.numeric.optimization.ObjectiveFunction;
import com.eggloop.flow.numeric.optimization.PointValue;
import com.eggloop.flow.numeric.optimization.methods.ConjugateGradientApache;
import com.eggloop.flow.numeric.optimization.methods.PowellMethodApache;
import com.eggloop.flow.numeric.optimization.methods.PowellMethodApacheConstrained;
import com.eggloop.flow.simhya.simhya.matlab.genetic.*;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.CMAESOptimizer;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.PowellOptimizer;

import java.util.Arrays;
import java.util.stream.IntStream;

public class Test2 {
    public static void main(String[] args) {
        GeneticOptions.setMax_time_bound(1);
        GeneticOptions.setMin_time_bound(0);
        int size=1;
        double Tf=1;
        FormulaPopulation pop = new FormulaPopulation(1);

        String[] variables = new String[]{"flow", "flow1"};
        double[] lower = new double[]{-10000, -500};
        double[] upper = new double[]{0, 500};
        for (int i = 0; i < variables.length; i++) {
            pop.addVariable(variables[i], lower[i], upper[i]);
        }
       //pop.generateInitialPopulation();
    //    pop.addFormula("flow <= Theta_57 U[Tl_58, Tu_58] flow >= Theta_56");
        String[] pars = new String[]{"Tl_1","Tu_1","Theta_2"};
        //double[] vv = new double[]{0.39, 1.67, -144};
        double[] vv = new double[]{0.39, 0.8, -4};
        String FF = "P=?[ F[Tl_1,Tu_1] {flow1 <= Theta_2} ]";
        //String FF = "P=?[{flow >= Theta_2018} U[Tl_2020, Tu_2020] {flow <= Theta_2019}]";

        Formula formula = pop.loadFormula(FF, pars, vv);
//        for (int i = 0; i <size ; i++) {
//            System.out.println(pop.getFormula(i).toString());
//        }

        NormalBreathSimulator normal_model = new NormalBreathSimulator(Test.class.getClassLoader().getResource("data/normal_model/").getPath());
        normal_model.setFinalExpirationPhaseOnly(true);
        normal_model.setSavePhase(false);
        IneffectiveBreathSimulator ineffective_model = new IneffectiveBreathSimulator(Test.class.getClassLoader().getResource("data/ineffective_model/").getPath());
        ineffective_model.setFinalExpirationPhaseOnly(true);
        ineffective_model.setSavePhase(false);

        continuousOptimization(pop, formula, normal_model, ineffective_model, Tf,1000);
        double[] p1u1 = smc(pop, formula, normal_model, Tf, 1000);
        double[] p2u2 = smc(pop, formula, ineffective_model, Tf, 1000);
        System.out.println(Arrays.toString(p1u1));
        System.out.println(Arrays.toString(p2u2));
        String[] bounds = formula.getTimeBounds();
        variables = formula.getVariables();
        String[] trshld = formula.getThresholds();
        String[] parsTrshld = new String[bounds.length + variables.length];
        System.arraycopy(bounds, 0, pars, 0, bounds.length);
        System.arraycopy(variables, 0, pars, bounds.length, variables.length);
        System.arraycopy(bounds, 0, parsTrshld, 0, bounds.length);
        System.arraycopy(trshld, 0, parsTrshld, bounds.length, variables.length);
        pop.setParameters(parsTrshld,new double[]{0.39,1.67,-144});
        p1u1 = smc(pop, formula, normal_model, Tf, 1000);
        p2u2 = smc(pop, formula, ineffective_model, Tf, 1000);
        System.out.println(Arrays.toString(p1u1));
        System.out.println(Arrays.toString(p2u2));



    }
    static double[] smc(FormulaPopulation popgen, Formula formula, BreathSimulator simulator, double Tf, int samples) {
        int[] data = popgen.modelCheck(simulator, formula, samples, Tf);
        double p = (double) (Arrays.stream(data).filter(x -> x == 1).count() + 1) / (samples + 2);
        double u = (double) (Arrays.stream(data).filter(x -> x == -1).count()) / (samples);
//        p = (sum(data==1)+1)/(samples+2);
//        u = (sum(data==-1))/(samples)
        return new double[]{p, u};

    }
    static void continuousOptimization(FormulaPopulation popgen, Formula formula, BreathSimulator simulator1, BreathSimulator simulator2, double Tf, int samples) {
        double[] timeBounds = new double[]{GeneticOptions.min_time_bound,GeneticOptions.max_time_bound};
        String[] bounds = formula.getTimeBounds();
        String[] variables = formula.getVariables();
        String[] trshld = formula.getThresholds();
        String[] pars = new String[bounds.length + variables.length];
        String[] parsTrshld = new String[bounds.length + variables.length];
        System.arraycopy(bounds, 0, pars, 0, bounds.length);
        System.arraycopy(variables, 0, pars, bounds.length, variables.length);
        System.arraycopy(bounds, 0, parsTrshld, 0, bounds.length);
        System.arraycopy(trshld, 0, parsTrshld, bounds.length, variables.length);
        double[] timeBoundsLb = Arrays.stream(bounds).mapToDouble(x -> GeneticOptions.min_time_bound).toArray();
        double[] timeBoundsUb = Arrays.stream(bounds).mapToDouble(x -> GeneticOptions.max_time_bound).toArray();
        double[] lbThrshld = Arrays.stream(variables).mapToDouble(popgen::getLowerBound).toArray();
        double[] ubThrshld = Arrays.stream(variables).mapToDouble(popgen::getUpperBound).toArray();
        double[] lb = new double[pars.length];
        double[] ub = new double[pars.length];
        System.arraycopy(timeBoundsLb, 0, lb, 0, timeBoundsLb.length);
        System.arraycopy(lbThrshld, 0, lb, bounds.length, lbThrshld.length);
        System.arraycopy(timeBoundsUb, 0, ub, 0, timeBoundsUb.length);
        System.arraycopy(ubThrshld, 0, ub, bounds.length, ubThrshld.length);
        //double[] start = IntStream.range(0, lb.length).mapToDouble(i -> (lb[i] + ub[i]) / 2).toArray();
        double[] start = popgen.getParameterValues(formula.getParameters());
        PowellMethodApache alg = new PowellMethodApache();
       // ConjugateGradientApache alg = new ConjugateGradientApache();
        ObjectiveFunction function = new FindBestParamtersFromSimulations2(popgen, simulator1, simulator2, Tf, formula, parsTrshld, samples, timeBounds);
        //PowellOptimizer optimizer = new PowellOptimizer();
        PointValue best = alg.optimise(function, start);
        //PointValue best = alg.optimise(function, start);
        double[] bestPar = best.getPoint();
        bestPar = reconvertTimeBounds(pars,bestPar,timeBounds);
        popgen.setParameters(parsTrshld,bestPar);

    }

    private static double[] reconvertTimeBounds(String[] parN, double[] newP, double[] timeBounds) {
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

    static void continuousOptimization2(FormulaPopulation popgen, Formula formula, BreathSimulator simulator1, BreathSimulator simulator2, double Tf, int samples) {
        double[] timeBounds = new double[]{GeneticOptions.min_time_bound, GeneticOptions.max_time_bound};
        String[] bounds = formula.getTimeBounds();
        String[] variables = formula.getVariables();
        String[] trshld = formula.getThresholds();
        String[] pars = new String[bounds.length + variables.length];
        String[] parsTrshld = new String[bounds.length + variables.length];
        System.arraycopy(bounds, 0, pars, 0, bounds.length);
        System.arraycopy(variables, 0, pars, bounds.length, variables.length);
        System.arraycopy(bounds, 0, parsTrshld, 0, bounds.length);
        System.arraycopy(trshld, 0, parsTrshld, bounds.length, variables.length);
        double[] timeBoundsLb = Arrays.stream(bounds).mapToDouble(x -> GeneticOptions.min_time_bound).toArray();
        double[] timeBoundsUb = Arrays.stream(bounds).mapToDouble(x -> GeneticOptions.max_time_bound).toArray();
        double[] lbThrshld = Arrays.stream(variables).mapToDouble(popgen::getLowerBound).toArray();
        double[] ubThrshld = Arrays.stream(variables).mapToDouble(popgen::getUpperBound).toArray();
        double[] lb = new double[pars.length];
        double[] ub = new double[pars.length];
        System.arraycopy(timeBoundsLb, 0, lb, 0, timeBoundsLb.length);
        System.arraycopy(lbThrshld, 0, lb, bounds.length, lbThrshld.length);
        System.arraycopy(timeBoundsUb, 0, ub, 0, timeBoundsUb.length);
        System.arraycopy(ubThrshld, 0, ub, bounds.length, ubThrshld.length);
        //double[] start = IntStream.range(0, lb.length).mapToDouble(i -> (lb[i] + ub[i]) / 2).toArray();
        double[] start = popgen.getParameterValues(formula.getParameters());
        double maxValue = -Double.MAX_VALUE;
        double[] bestNp = Arrays.copyOf(start,start.length);
        for (int i = 0; i < 100; i++) {
            System.out.println(i);
//            double[] ran= Arrays.stream(lb).map(v1 -> v1 + Math.random()).toArray();
//            double[] nP= IntStream.range(0, lb.length).mapToDouble(k -> k%2==1&&k<bounds.length ? ran[k-1]+(ub[k]-ran[k-1])*Math.random():k).toArray();

            double[] par = IntStream.range(0, lb.length).mapToDouble(k -> lb[k] + Math.random() * (ub[k] - lb[k])).toArray();
            double[] nP = reconvertTimeBounds(parsTrshld, par, timeBounds);
            int[] data1 = popgen.modelCheck(simulator1, formula, samples, Tf);
            int[] data2 = popgen.modelCheck(simulator2, formula, samples, Tf);
            double v = fitnessBootstrap(formula, data1, data2);
            if (v > maxValue) {
                maxValue = v;
                bestNp = nP;
                System.out.println(maxValue);
                System.out.println(Arrays.toString(bestNp));
            }
        }
        popgen.setParameters(parsTrshld,bestNp);
    }
    private static double fitnessBootstrap(Formula formula, int[] data1, int[] data2) {
        int N = data1.length;
        double p1 = (double)(Arrays.stream(data1).filter(x -> x == 1).count());
        double p2 = (double)(Arrays.stream(data2).filter(x -> x == 1).count());
        double u1 = (double)Arrays.stream(data1).filter(x -> x == -1).sum() / (double)(N);
        double u2 = (double)Arrays.stream(data2).filter(x -> x == -1).sum() / (double)(N);
        FitnessFunction fitness = new RegularisedLogOddRatioFitness();
        return fitness.compute(p1, p2, formula.getFormulaSize(), u1, u2, N);
      //  return (p1+p2);
    }
}
