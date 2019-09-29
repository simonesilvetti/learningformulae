package com.eggloop.flow.simhya.simhya.matlab;

import com.eggloop.flow.simhya.simhya.matheval.SymbolArray;
import com.eggloop.flow.simhya.simhya.matlab.genetic.Formula;
import com.eggloop.flow.simhya.simhya.matlab.genetic.FormulaGenOps;
import com.eggloop.flow.simhya.simhya.matlab.genetic.FormulaPopulation;
import com.eggloop.flow.simhya.simhya.matlab.genetic.GeneticOptions;
import com.eggloop.flow.simhya.simhya.modelchecking.mtl.*;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

public class FINALE_PROVA_INIT_GENERATION {
    public static void main(String[] args) {
        int N = 80;
        FormulaPopulation pop = new FormulaPopulation(N);
        String[] variables = new String[]{"x", "y"};
        double[] lower = new double[]{0, 0};
        double[] upper = new double[]{80, 45};
        for (int i = 0; i < variables.length; i++) {
            pop.addVariable(variables[i], lower[i], upper[i]);
        }
        pop.addGeneticInitFormula(pop.getVariableNumber());

        FormulaGenOps ops = new FormulaGenOps(pop, new SymbolArray());
        List<MTLnode> atomicNode = combineAtomic(pop,ops);
        List<MTLnode> Ornode = applyOr(atomicNode,ops);
        List<MTLnode> Andnode = applyAnd(atomicNode);

        List<MTLnode> Fnode = applyFinally(atomicNode, ops);
        List<MTLnode> Gnode = applyGlobally(atomicNode, ops);
        List<MTLnode> GOrnode = applyGlobally(Ornode, ops);
        List<MTLnode> FAndnode = applyFinally(Andnode, ops);


        //FormulaGenOps ops = new FormulaGenOps(pop, new SymbolArray());
        List<MTLnode> initSet= new ArrayList<>();

        initSet.addAll(Fnode);
        initSet.addAll(Gnode);
        initSet.addAll(GOrnode);
        initSet.addAll(FAndnode);

        List<Formula> initFormula= new ArrayList<>();
        for (MTLnode mtLnode : initSet) {
            initFormula.add(new Formula(new MTLformula(mtLnode)));
        }

        FormulaPopulation pop2 = new FormulaPopulation(N);
        String[] variables2 = new String[]{"x", "y"};
        double[] lower2 = new double[]{0, 0};
        double[] upper2 = new double[]{80, 45};
        for (int i = 0; i < variables.length; i++) {
            pop2.addVariable(variables[i], lower[i], upper[i]);
        }
        FormulaGenOps ops2 = new FormulaGenOps(pop, new SymbolArray());
        pop2.loadFormula(initFormula.get(1).toString(),initFormula.get(1).getParameters(),new double[initFormula.get(1).getParameters().length]);







        GeneticOptions.init__globallyeventually_weight = 0.05;
        GeneticOptions.init__eventuallyglobally_weight = 0.05;
//        GeneticOptions.init__globally_weight = 0.7;
//        GeneticOptions.init__eventually_weight = 0.7;
//        GeneticOptions.init__not_weight=0;
//        GeneticOptions.init__until_weight = 0.9;
        //GeneticOptions.setInit__fixed_number_of_atoms(4);
        //GeneticOptions.setInit__prob_of_true_atom(0);
        GeneticOptions.setMin_time_bound(0);
        GeneticOptions.setMax_time_bound(300);
        pop.generateInitialPopulation();



        for (int i = 0; i < variables.length; i++) {
            MTLnode mtLnode = ops.newAtomicNodeCustom(false, i, true);
            MTLformula f = new MTLformula(mtLnode);
            System.out.println(f.toString());
            mtLnode = ops.newAtomicNodeCustom(false, i, false);
            f = new MTLformula(mtLnode);
            System.out.println(f.toString());
        }

        List<int[]> ints = combinatorics2(3);
        System.out.println("a");


//        ops.randomFormula();
//        for (int i = 0; i < pop.getPopulationSize(); i++) {
//            System.out.println(pop.getFormula(i).toString());
//        }
    }

    public static List<int[]> combinatorics2(int n) {
        List<int[]> res = new ArrayList<>();
        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                res.add(new int[]{i, j});
                res.add(new int[]{j, i});
            }
        }
        return res;

    }


    static List<MTLnode> combineAtomic(FormulaPopulation pop, FormulaGenOps ops) {
        List<MTLnode> res = new ArrayList<>();
        int n = pop.getVariableNumber();
        for (int i = 0; i < n; i++) {
            MTLnode mtLnode = ops.newAtomicNodeCustom(false, i, true);
            //MTLformula f = new MTLformula(mtLnode);
            res.add(mtLnode);
            mtLnode = ops.newAtomicNodeCustom(false, i, false);
            //f = new MTLformula(mtLnode);
            res.add(mtLnode);
        }
        return res;
    }

    static List<MTLnode> applyFinally(List<MTLnode> formulaSet,FormulaGenOps ops) {
        List<MTLnode> res = new ArrayList<>();
        for (MTLnode mtlNode : formulaSet) {
            res.add(Finally(mtlNode,ops));
        }
    return res;
    }

    static List<MTLnode> applyGlobally(List<MTLnode> formulaSet,FormulaGenOps ops) {
        List<MTLnode> res = new ArrayList<>();
        for (MTLnode mtlNode : formulaSet) {
            res.add(Globally(mtlNode,ops));
        }
        return res;
    }

    static List<MTLnode> applyAnd(List<MTLnode> formulaSet) {
        List<MTLnode> res = new ArrayList<>();
        List<int[]> ints = combinatorics2(formulaSet.size());
        for (int[] anInt : ints) {
            res.add(new MTLand(formulaSet.get(anInt[0]).duplicate(),formulaSet.get(anInt[1]).duplicate()));
        }
        return res;
    }

    static List<MTLnode> applyOr(List<MTLnode> formulaSet,FormulaGenOps ops) {
        List<MTLnode> res = new ArrayList<>();
        List<int[]> ints = combinatorics2(formulaSet.size());
        for (int[] anInt : ints) {
            MTLnode duplicate = formulaSet.get(anInt[0]);
            MTLformula f = new MTLformula(duplicate);
            Formula duplicate2 = ops.duplicate(f);
            MTLnode root = f.getRoot();


            MTLnode duplicate1 = formulaSet.get(anInt[1]);
            res.add(new MTLor(duplicate, duplicate1));
        }
      return res;
    }


    private static MTLnode Finally(MTLnode mtlNode,FormulaGenOps ops) {

        long id = MTLnode.getNextID();
        ParametricInterval interval = ops.newInterval(id);
        return new MTLeventually(interval,mtlNode);
    }

    private static MTLnode Globally(MTLnode mtlNode,FormulaGenOps ops) {
        long id = MTLnode.getNextID();
        ParametricInterval interval = ops.newInterval(id);
        return new MTLglobally(interval,mtlNode);
    }





}





