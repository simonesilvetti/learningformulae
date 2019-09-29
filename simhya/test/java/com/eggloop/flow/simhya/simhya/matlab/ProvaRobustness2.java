package com.eggloop.flow.simhya.simhya.matlab;

import com.eggloop.flow.expr.Context;
import com.eggloop.flow.expr.Variable;
import com.eggloop.flow.gaussianprocesses.gp.kernels.KernelRBF;
import com.eggloop.flow.gpoptimisation.gpoptim.GPOptimisation;
import com.eggloop.flow.gpoptimisation.gpoptim.GpoOptions;
import com.eggloop.flow.gpoptimisation.gpoptim.GpoResult;
import com.eggloop.flow.mitl.MiTL;
import com.eggloop.flow.mitl.MitlPropertiesList;
import com.eggloop.flow.model.Trajectory;
import com.eggloop.flow.numeric.optimization.ObjectiveFunction;
import com.eggloop.flow.parsers.MitlFactory;
import com.eggloop.flow.sampler.GridSampler;
import com.eggloop.flow.sampler.Parameter;
import com.eggloop.flow.simhya.simhya.matlab.genetic.Formula;
import com.eggloop.flow.simhya.simhya.matlab.genetic.FormulaPopulation;
import com.eggloop.flow.simhya.simhya.matlab.genetic.GeneticOptions;
import com.eggloop.flow.utils.data.TrajectoryReconstruction;
import com.eggloop.flow.utils.files.Utils;

import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

public class ProvaRobustness2 {

    public static void main(String[] args) {
        Random ran = new Random(1);
        double[][] ds2SpatialValues = Utils.readMatrixFromFile(TestReading.class.getClassLoader().getResource("data/calin/ds2Trajectories").getPath());
        double[] ds2Labels = Utils.readVectorFromFile(TestReading.class.getClassLoader().getResource("data/calin/ds2Labels").getPath());
        double[] ds2Times = Utils.readVectorFromFile(TestReading.class.getClassLoader().getResource("data/calin/ds2Times").getPath());
        TrajectoryReconstruction data = new TrajectoryReconstruction(ds2Times, ds2SpatialValues, ds2Labels, 0.5, ran);
        data.split();
        double[][] normal_model = data.getPoistiveTrainSet();
        double[][] ineffective_model = data.getNegativeTrainSet();
        double[][] normal_model_test = data.getPoistiveValidationSet();
        double[][] ineffective_model_test = data.getNegativeValidationSet();


        FormulaPopulation pop = new FormulaPopulation(1);
        String[] variables = new String[]{"flow"};
        double[] lower = new double[]{0};
        double[] upper = new double[]{12};
        for (int i = 0; i < variables.length; i++) {
            pop.addVariable(variables[i], lower[i], upper[i]);
        }
        String[] pars = new String[]{"Tl_3", "Tu_3", "Theta_3"};
        double[] vv = new double[]{0, 53, 9.25};
        String FF = "P=?[F[Tl_3,Tu_3] {flow >= Theta_3}]";
///(G[Tl_1, Tu_1] ((flow <= Theta_1 & G[Tl_2, Tu_2] (flow <= Theta_2) ))  | F[Tl_3, Tu_3] flow >= Theta_3 )
//       (G[Tl_1,Tu_1] {flow<=Theta_1} AND  G[Tl_2,Tu_2] {flow<=Theta_2}) OR  F[Tl_3,Tu_3] {flow >= Theta_3}
//        ((G[Tl_1, Tu_1] ((flow <= Theta_1 & (G[Tl_2, Tu_2] (flow <= Theta_2)) )))  | (F[Tl_3, Tu_3] flow >= Theta_3))
        // ((G[Tl_1, Tu_1] (flow <= Theta_1 & (G[Tl_2, Tu_2] flow <= Theta_2) ))  | (F[Tl_3, Tu_3] flow >= Theta_3))
        //    (G[Tl_1, Tu_1] ((flow <= Theta_1 & G[Tl_2, Tu_2] (flow <= Theta_2) ))  | F[Tl_3, Tu_3] (flow >= Theta_3))
        pars = new String[]{"Tl_1", "Tu_1", "Tl_2", "Tu_2", "Tl_3", "Tu_3", "Theta_1", "Theta_2", "Theta_3"};
        vv = new double[]{0.367, 52.2, 47, 96.6, 0.367, 52.2, 9.31, 9, 9.31};
        FF = "P=?[ (G[Tl_1,Tu_1] {flow<=Theta_1} AND  G[Tl_2,Tu_2] {flow<=Theta_2}) OR  F[Tl_3,Tu_3] {flow >= Theta_3} ]";
        Formula formula = pop.loadFormula(FF, pars, vv);
        System.out.println(formula.toString());
        //char[] a=formula.toSign().toCharArray();
        //System.out.println(formula.toString());

   //     double [] val = findParamterClassification(variables,new double []{GeneticOptions.min_time_bound,GeneticOptions.max_time_bound},ds2Times,normal_model,ineffective_model,formula,pop);

        double[] timeBoundsFormula = new double[]{GeneticOptions.min_time_bound, GeneticOptions.max_time_bound};
        String[] boundsFormula = formula.getTimeBounds();
        String[] variablesFormula = formula.getVariables();
        String[] thresholdFormula = formula.getThresholds();
        double[] timeBoundsLb = Arrays.stream(boundsFormula).mapToDouble(x -> GeneticOptions.min_time_bound).toArray();
        double[] timeBoundsUb = Arrays.stream(boundsFormula).mapToDouble(x -> GeneticOptions.max_time_bound).toArray();
        double[] thrshldLb = Arrays.stream(variables).mapToDouble(pop::getLowerBound).toArray();
        double[] thrshldUb = Arrays.stream(variables).mapToDouble(pop::getUpperBound).toArray();
        double[] lb = new double[pars.length];
        double[] ub = new double[pars.length];
        System.arraycopy(timeBoundsLb, 0, lb, 0, timeBoundsLb.length);
        System.arraycopy(thrshldLb, 0, lb, boundsFormula.length, thrshldLb.length);
        System.arraycopy(timeBoundsUb, 0, ub, 0, timeBoundsUb.length);
        System.arraycopy(thrshldUb, 0, ub, boundsFormula.length, thrshldUb.length);


        ObjectiveFunction function = point -> {
            for (int i = 0; i < boundsFormula.length; i += 2) {
                point[i + 1] = point[i] + point[i + 1] * (1 - point[i]);
            }
            final double[] p = point;
            point = IntStream.range(0, point.length).mapToDouble(i -> lb[0] + p[i] * (ub[1] - lb[0])).toArray();
            double[] value1 = computeAverageRobustness(ds2Times, normal_model, variables, formula, point);
            double[] value2 = computeAverageRobustness(ds2Times, ineffective_model, variables, formula,  point);
            //double abs = Math.max(value1[3]-value2[2],value2[3]-value1[2]);
            //return distanceSet(value1[2],value1[3],value2[2],value2[3]);
            //double abs = Math.abs((value1[0] - value2[0]) / (3 * (value1[1] + value2[1])));
            double abs = (value1[0] - value2[0]) / (3 * (value1[1] + value2[1]));
            System.out.println(Arrays.toString(value1)+"||"+Arrays.toString(value2)+":"+abs);
            return abs;

        };

        GridSampler custom = new GridSampler() {
            @Override
            public double[][] sample(int n, double[] lbounds, double[] ubounds) {
                double[][] res = new double[n][lbounds.length];
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < boundsFormula.length; j += 2) {
                        res[i][j] = lbounds[j] + Math.random() * (ubounds[j] - lbounds[j]);
                        res[i][j + 1] = res[i][j] + Math.random() * (ubounds[j] - res[i][j]);
                    }
                    for (int j = boundsFormula.length; j < res[i].length; j++) {
                        res[i][j] = lbounds[j] + Math.random() * (ubounds[j] - lbounds[j]);
                    }
                }
                return res;
            }

            @Override
            public double[][] sample(int n, Parameter[] params) {
                return new double[0][];
            }
        };

        GPOptimisation gpo = new GPOptimisation();
        GpoOptions options = new GpoOptions();
        options.setInitialSampler(custom);
        //options.setGridSampler(new LatinHypercubeSampler(2, 10));
        options.setMaxIterations(800);
        options.setHyperparamOptimisation(true);
        options.setUseNoiseTermRatio(false);
        options.setGridSampler(custom);
        options.setInitialObservations(500);
        //options.setMaxAddedPointsNoImprovement(300);
        //options.setMaxFailedAttempts(300);
        KernelRBF kernelGP = new KernelRBF();
        //kernelGP.setHyperarameters(new double[]{Math.log(normal_model.length + 1), 0.4});
        options.setKernelGP(kernelGP);
        //options.setGridSampleNumber(200);
        gpo.setOptions(options);
        GpoResult optimise;
        double[] lbU = IntStream.range(0, lb.length ).mapToDouble(i -> 0).toArray();
        double[] ubU = IntStream.range(0, ub.length ).mapToDouble(i -> 1).toArray();
        optimise = gpo.optimise(function, lbU, ubU);
        // double[] v = optimise.getSolution();
        //System.out.println(Arrays.toString(v));
       // final double[] v = concatenate(optimise.getSolution(), new double[]{0});
        final double[] v = optimise.getSolution();
        //vv= Arrays.stream(vv).map(v -> lb[0] + v * (ub[1] - lb[0])).toArray();
        vv = IntStream.range(0, vv.length).mapToDouble(i -> lb[i] + v[i] * (ub[i] - lb[i])).toArray();
        System.out.println(optimise.getFitness()+"OTTIMO:"+Arrays.toString(vv));
        //System.out.println("àà"+Arrays.toString(vv));
        //System.out.println(Arrays.toString(optimise.getSolution()));

        double[] p1u1 = computeAverageRobustness(ds2Times, normal_model, variables, formula, vv);
        System.out.println("OLD:" + Arrays.toString(p1u1));
        double[] p2u2 = computeAverageRobustness(ds2Times, ineffective_model, variables, formula, vv);
        System.out.println("OLD:" + Arrays.toString(p2u2));
        double value;
        if (p1u1[0] > p2u2[0]) {
            value = ((p1u1[0] - 3 * p1u1[1]) + (p2u2[0] + 3 * p2u2[1])) / 2;
        } else {
            value = ((p2u2[0] - 3 * p2u2[1]) + (p1u1[0] + 3 * p1u1[1])) / 2;
        }
        // double value = (p1u1[0]+p2u2[0])/2;
        char[] a = formula.toSign().toCharArray();
        System.out.println(a.length);
        System.out.println(vv.length - timeBoundsLb.length);
        for (int i = timeBoundsLb.length; i < vv.length; i++) {
            if (a[i - timeBoundsLb.length] == '1') {
                vv[i] += value;
            } else {
                vv[i] = Math.max(vv[i] - value, 0);
            }
        }

        System.out.println(Arrays.toString(formula.getParameters()));
        System.out.println(Arrays.toString(vv));
        p1u1 = computeAverageRobustness(ds2Times, normal_model_test, variables, formula, vv);
        System.out.println(Arrays.toString(p1u1));
        p2u2 = computeAverageRobustness(ds2Times, ineffective_model_test, variables, formula, vv);
        System.out.println(Arrays.toString(p2u2));

        p1u1 = smcNew(ds2Times, normal_model_test, variables, formula, vv);
        System.out.println("Noi:" + Arrays.toString(p1u1));
        p2u2 = smcNew(ds2Times, ineffective_model_test, variables, formula, vv);
        System.out.println("Noi:" + Arrays.toString(p2u2));

        //(((G_{[0.367,52.2)}x_{1}<9.31) \wedge (G_{[47,96.6)}x_{1}<9) ) \vee (F_{[0.367,52.2)}x_{1}>9.31) )


        vv = new double[]{0.367, 52.2, 47, 96.6, 0.367, 52.2, 9.31, 9, 9.31};
        System.out.println(Arrays.toString(vv));
        p1u1 = smcNew(ds2Times, normal_model_test, variables, formula, vv);
        System.out.println("CALIN_08:" + Arrays.toString(p1u1));
        p2u2 = smcNew(ds2Times, ineffective_model_test, variables, formula, vv);
        System.out.println("CALIN_08:" + Arrays.toString(p2u2));

        //(((G_{[17.4,44.2)}x_{1}<9.3) \wedge (G_{[52.8,77.3)}x_{1}<9.05) ) \vee (F_{[17.4,44.2)}x_{1}>9.3) )

        vv = new double[]{17.4, 44.2, 52.8, 77.3, 17.4, 44.2, 9.3, 9.05, 9.3};
        System.out.println(Arrays.toString(vv));
        p1u1 = smcNew(ds2Times, normal_model_test, variables, formula, vv);
        System.out.println("CALIN_06:" + Arrays.toString(p1u1));
        p2u2 = smcNew(ds2Times, ineffective_model_test, variables, formula, vv);
        System.out.println("CALIN_06:" + Arrays.toString(p2u2));



    }

    public static double[] findParamterClassification(String[] variables,double[] timebounds,double[] ds2Times,double[][] normal_model,double[][] ineffective_model,Formula formula, FormulaPopulation pop){

        double[] timeBoundsFormula = new double[]{timebounds[0], timebounds[1]};
        String[] boundsFormula = formula.getTimeBounds();
        String[] variablesFormula = formula.getVariables();
        String[] thresholdFormula = formula.getThresholds();
        double[] timeBoundsLb = Arrays.stream(boundsFormula).mapToDouble(x -> GeneticOptions.min_time_bound).toArray();
        double[] timeBoundsUb = Arrays.stream(boundsFormula).mapToDouble(x -> GeneticOptions.max_time_bound).toArray();
        double[] thrshldLb = Arrays.stream(variables).mapToDouble(pop::getLowerBound).toArray();
        double[] thrshldUb = Arrays.stream(variables).mapToDouble(pop::getUpperBound).toArray();
        double[] lb = new double[formula.getParameters().length];
        double[] ub = new double[formula.getParameters().length];
        System.arraycopy(timeBoundsLb, 0, lb, 0, timeBoundsLb.length);
        System.arraycopy(thrshldLb, 0, lb, boundsFormula.length, thrshldLb.length);
        System.arraycopy(timeBoundsUb, 0, ub, 0, timeBoundsUb.length);
        System.arraycopy(thrshldUb, 0, ub, boundsFormula.length, thrshldUb.length);


        ObjectiveFunction function = point -> {
            for (int i = 0; i < boundsFormula.length; i += 2) {
                point[i + 1] = point[i] + point[i + 1] * (1 - point[i]);
            }
            final double[] p = point;
            point = IntStream.range(0, point.length).mapToDouble(i -> lb[0] + p[i] * (ub[1] - lb[0])).toArray();
            double[] value1 = computeAverageRobustness(ds2Times, normal_model, variables, formula, point);
            double[] value2 = computeAverageRobustness(ds2Times, ineffective_model, variables, formula,  point);
            double abs = Math.max(value1[3]-value2[2],value2[3]-value1[2]);
            //return distanceSet(value1[2],value1[3],value2[2],value2[3]);
            //double abs = Math.abs((value1[0] - value2[0]) / (3 * (value1[1] + value2[1])));
            //double abs = (value1[0] - value2[0]) / (3 * (value1[1] + value2[1]));
            //System.out.println(Arrays.toString(value1)+"||"+Arrays.toString(value2)+":"+abs);
            return abs;

        };

        GridSampler custom = new GridSampler() {
            @Override
            public double[][] sample(int n, double[] lbounds, double[] ubounds) {
                double[][] res = new double[n][lbounds.length];
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < boundsFormula.length; j += 2) {
                        res[i][j] = lbounds[j] + Math.random() * (ubounds[j] - lbounds[j]);
                        res[i][j + 1] = res[i][j] + Math.random() * (ubounds[j] - res[i][j]);
                    }
                    for (int j = boundsFormula.length; j < res[i].length; j++) {
                        res[i][j] = lbounds[j] + Math.random() * (ubounds[j] - lbounds[j]);
                    }
                }
                return res;
            }

            @Override
            public double[][] sample(int n, Parameter[] params) {
                return new double[0][];
            }
        };

        GPOptimisation gpo = new GPOptimisation();
        GpoOptions options = new GpoOptions();
        options.setInitialSampler(custom);
        //options.setGridSampler(new LatinHypercubeSampler(2, 10));
        options.setMaxIterations(800);
        options.setHyperparamOptimisation(true);
        options.setUseNoiseTermRatio(false);
        options.setGridSampler(custom);
        options.setInitialObservations(300);
        //options.setMaxAddedPointsNoImprovement(300);
        //options.setMaxFailedAttempts(300);
        KernelRBF kernelGP = new KernelRBF();
        //kernelGP.setHyperarameters(new double[]{Math.log(normal_model.length + 1), 0.4});
        options.setKernelGP(kernelGP);
        //options.setGridSampleNumber(200);
        gpo.setOptions(options);
        GpoResult optimise;
        double[] lbU = IntStream.range(0, lb.length ).mapToDouble(i -> 0).toArray();
        double[] ubU = IntStream.range(0, ub.length ).mapToDouble(i -> 1).toArray();
        optimise = gpo.optimise(function, lbU, ubU);
        // double[] v = optimise.getSolution();
        //System.out.println(Arrays.toString(v));
        // final double[] v = concatenate(optimise.getSolution(), new double[]{0});
        final double[] v = optimise.getSolution();
        //vv= Arrays.stream(vv).map(v -> lb[0] + v * (ub[1] - lb[0])).toArray();
        double [] vv = IntStream.range(0, v.length).mapToDouble(i -> lb[i] + v[i] * (ub[i] - lb[i])).toArray();
        System.out.println(optimise.getFitness()+"OTTIMO:"+Arrays.toString(vv));
        //System.out.println("àà"+Arrays.toString(vv));
        //System.out.println(Arrays.toString(optimise.getSolution()));

        double[] p1u1 = computeAverageRobustness(ds2Times, normal_model, variables, formula, vv);
        System.out.println("OLD:" + Arrays.toString(p1u1));
        double[] p2u2 = computeAverageRobustness(ds2Times, ineffective_model, variables, formula, vv);
        System.out.println("OLD:" + Arrays.toString(p2u2));
        double value;
        if (p1u1[0] > p2u2[0]) {
            value = ((p1u1[0] - 3 * p1u1[1]) + (p2u2[0] + 3 * p2u2[1])) / 2;
        } else {
            value = ((p2u2[0] - 3 * p2u2[1]) + (p1u1[0] + 3 * p1u1[1])) / 2;
        }
        // double value = (p1u1[0]+p2u2[0])/2;
        char[] a = formula.toSign().toCharArray();
        System.out.println(a.length);
        System.out.println(vv.length - timeBoundsLb.length);
        for (int i = timeBoundsLb.length; i < vv.length; i++) {
            if (a[i - timeBoundsLb.length] == '1') {
                vv[i] += value;
            } else {
                vv[i] = Math.max(vv[i] - value, 0);
            }
        }

     return vv;

    }

    public static double distanceSet(double a1,double a2,double b1, double b2){
        double v = Math.max(Math.abs(a1-b2),Math.abs(b1-a2));
        return v/(a2-a1+b2-b1);
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
        //builder.append("((G[Tl_1, Tu_1] (flow <= Theta_1) & G[Tl_2, Tu_2] (flow <= Theta_2) )  | F[Tl_3, Tu_3] (flow >= Theta_3))\n");
        // builder.append("((G[Tl_1, Tu_1] (flow <= Theta_1) & G[Tl_2, Tu_2] (flow <= Theta_2))  | F[Tl_3, Tu_3] (flow >= Theta_3))\n");
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
        double sigma = Math.sqrt(Arrays.stream(b).map(x -> (x - mean) * (x - mean)).sum() / b.length);
        double min = Arrays.stream(b).min().getAsDouble();
        double max = Arrays.stream(b).max().getAsDouble();


        return new double[]{mean, sigma,min,max};

    }

    static double[] smc(FormulaPopulation popgen, Formula formula, double[] parameters, double[] times, double[][] trajectories) {
        int[] data = new int[trajectories.length];
        for (int i = 0; i < trajectories.length; i++) {
            popgen.setParameters(formula.getParameters(), parameters);
            data[i] = popgen.modelCheck(combineTimeAndSpace(times, new double[][]{trajectories[i]}), formula);
        }
        double p = (double) (Arrays.stream(data).filter(x -> x == 1).count() + 1) / (trajectories.length + 2);
        double u = (double) (Arrays.stream(data).filter(x -> x == -1).count()) / (trajectories.length);
//        p = (sum(data==1)+1)/(samples+2);
//        u = (sum(data==-1))/(samples)
        return new double[]{p, u};

    }

    static double[] smcNew(double[] times, double[][] simulate, String[] variables, Formula formula, double[] vv) {
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
        //builder.append("((G[Tl_1, Tu_1] (flow <= Theta_1) & G[Tl_2, Tu_2] (flow <= Theta_2) )  | F[Tl_3, Tu_3] (flow >= Theta_3))\n");
        // builder.append("((G[Tl_1, Tu_1] (flow <= Theta_1) & G[Tl_2, Tu_2] (flow <= Theta_2))  | F[Tl_3, Tu_3] (flow >= Theta_3))\n");
        MitlFactory factory = new MitlFactory(ns);
        String text = builder.toString();
        //System.out.println(text);
        MitlPropertiesList l = factory.constructProperties(text);
        MiTL prop = l.getProperties().get(0);

        for (int i = 0; i < simulate.length; i++) {
            Trajectory x = new Trajectory(times, ns, new double[][]{simulate[i]});
            b[i] = prop.evaluate(x, 0) ? 1 : 0;
        }
        double mean = Arrays.stream(b).sum() / b.length;
        double variance = Arrays.stream(b).map(x -> (x - mean) * (x - mean)).sum() / b.length;
        return new double[]{mean, variance};


    }

    private static double[][] combineTimeAndSpace(double[] times, double[][] spaceTrajectories) {
        double[][] res = new double[spaceTrajectories.length + 1][];
        res[0] = times;
        System.arraycopy(spaceTrajectories, 0, res, 1, spaceTrajectories.length);
        return res;

    }
}

