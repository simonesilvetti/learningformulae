package com.eggloop.flow.simhya.simhya.matlab;

import com.eggloop.flow.simhya.simhya.matlab.genetic.Formula;
import com.eggloop.flow.simhya.simhya.matlab.genetic.FormulaPopulation;
import com.eggloop.flow.simhya.simhya.matlab.genetic.GeneticOptions;

import java.util.HashMap;
import java.util.Map;

public class TestGenetic {
    public static void main(String[] args) {
        GeneticOptions.setMin_time_bound(0);
        GeneticOptions.setMax_time_bound(100);
        GeneticOptions.setUndefined_reference_threshold(0.1);
        GeneticOptions.setSize_penalty_coefficient(1);
        GeneticOptions.setMutate__one_node(true);
        GeneticOptions.setMutate__mutation_probability_per_node(0.5);
        //GeneticOptions.setInit__fixed_number_of_atoms(2);
        GeneticOptions.setFitness_type("regularised_logodds");
        //GeneticOptions.init__random_number_of_atoms=true;
        //GeneticOptions.init__fixed_number_of_atoms=2;

        //TODO: SAGGIUSTARE LA SCALE DA MATLAB
        int N = 10;
        int runs = 1000;
        int Tf = 100;
        FormulaPopulation pop  = new FormulaPopulation(N);

//        fitnessOptions.type = 1; %0=normal, 1=modified
//        fitnessOptions.urf = GeneticOptions.undefined_reference_threshold;
//        fitnessOptions.spc = GeneticOptions.size_penalty_coefficient;
//        fitnessOptions.scale = 10;

        Map<Formula,double[]> formulaToParamters = new HashMap<>();
        Map<Formula,Double> formulaToPerformance = new HashMap<>();

        String[] variables = new String[]{"flow"};
        double[] lower = new double[]{0};
        double[] upper = new double[]{12};
        for (int i = 0; i < variables.length; i++) {
            pop.addVariable(variables[i], lower[i], upper[i]);
        }
        pop.generateInitialPopulation();
//        for (int i = 0; i < N; i++) {
//            double[] paramters = continuousOptimization(pop, pop.getFormula(i), normal_model, ineffective_model, ds2Times);
//            double p1= smc(pop, pop.getFormula(i),ds2Times, ineffective_model);
//            double p2 = smc(pop, pop.getFormula(i),ds2Times, normal_model);
//            double performance = fitnessScore[iterationIndex][i] = fitness.compute(p1u1[iterationIndex][i][0], p2u2[iterationIndex][i][0], pop.getFormula(i).getFormulaSize(), p1u1[iterationIndex][i][1], p2u2[iterationIndex][i][1], runs);
//        }

        pop.initialiseNewGeneration();
        for (int i = 0; i < N-1; i++) {
            pop.selectFormula(i);
            pop.selectFormula(i+1);
            pop.crossoverNewGeneration(i,i+1);
        }
        pop.finaliseNewGeneration();

        //pop.selectFormula(2);
        //pop.crossoverNewGeneration(1,2);
        //pop.mutateNewGeneration(0);


        System.out.println("___________");
        for (int i = 0; i < pop.getPopulationSize(); i++) {
            System.out.println(pop.getFormula(i).toString());
        }


    }
}
