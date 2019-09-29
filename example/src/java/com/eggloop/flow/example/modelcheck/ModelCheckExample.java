package com.eggloop.flow.example.modelcheck;

import com.eggloop.flow.expr.Context;
import com.eggloop.flow.expr.Variable;
import com.eggloop.flow.learning.Learning;
import com.eggloop.flow.mitl.MiTL;
import com.eggloop.flow.mitl.MitlPropertiesList;
import com.eggloop.flow.model.Trajectory;
import com.eggloop.flow.parsers.MitlFactory;
import com.eggloop.flow.utils.files.Utils;
import com.eggloop.flow.utils.string.StringUtils;

import java.util.Arrays;
import java.util.function.UnaryOperator;

public class ModelCheckExample {

    private static final UnaryOperator<String> FILE_PATH = Utils.getFilePath(Learning.class);
    private static double[] ds2Times = Utils.readVectorFromFile(FILE_PATH.apply("temporal/synthTime_12.txt"));
    private static double[][][] ds2SpatialValues = Utils.readMatrixMultiFromFile(ds2Times.length, FILE_PATH.apply("temporal/synthData_12.txt"));

    public static void main(String[] args) {
        String[] variables = new String[]{"y", "z"};
        String[] parameters = new String[]{"Tl_176", "Tu_176", "Theta_3", "Theta_6"};
        double[] parametersValues = new double[]{11.0, 11.0, 48.990792215517644, 37.53885223004586};
        String formulaPSTL = "G[Tl_176, Tu_176] ((y >= Theta_3 | z <= Theta_6))";
        String formulaSTL = StringUtils.replace(formulaPSTL, parameters, parametersValues) + "\n";
        Context variableContext = new Context();
        for (String s : variables) {
            new Variable(s, variableContext);
        }
        MitlFactory factory = new MitlFactory(variableContext);
        MitlPropertiesList l = factory.constructProperties(formulaSTL);
        MiTL prop = l.getProperties().get(0);

        boolean isSatisfied = qualitativeSemantics(0, variableContext, prop, ds2Times, ds2SpatialValues[0]);
        double robustness = quantiativeSemantics(0, variableContext, prop, ds2Times, ds2SpatialValues[0]);
        System.out.println("is satisfied? " + isSatisfied);
        System.out.println("robusntess? " + robustness);
    }

    private static double quantiativeSemantics(double t, Context variableContext, MiTL formula, double[] times, double[][] trajectory) {
        Trajectory x = new Trajectory(times, variableContext, trajectory);
        return formula.evaluateValue(x, t);
    }

    private static boolean qualitativeSemantics(double t, Context variableContext, MiTL formula, double[] times, double[][] trajectory) {
        Trajectory x = new Trajectory(times, variableContext, trajectory);
        return formula.evaluate(x, t);
    }

    private static int check(double[] times, double[][][] trajectories, String[] variables, String[] parameters, String formula, double[] formulaParameters, double atTime) {
        int[] b = new int[trajectories.length];
        Context ns = new Context();
        for (String s : variables) {
            new Variable(s, ns);
        }
        StringBuilder builder = new StringBuilder();
        for (int j = 0; j < parameters.length; j++) {
            builder.append("const double ").append(parameters[j]).append("=").append(formulaParameters[j]).append(";\n");
        }
        builder.append(formula).append("\n");
        MitlFactory factory = new MitlFactory(ns);
        String text = builder.toString();
        MitlPropertiesList l = factory.constructProperties(text);
        MiTL prop = l.getProperties().get(0);
        for (int i = 0; i < trajectories.length; i++) {
            Trajectory x = new Trajectory(times, ns, trajectories[i]);
            b[i] = prop.evaluate(x, atTime) ? 1 : 0;
        }
        return Arrays.stream(b).sum();
    }

}
