package com.eggloop.flow.parsers;

import com.eggloop.flow.expr.Context;
import com.eggloop.flow.expr.Variable;
import com.eggloop.flow.mitl.MiTL;
import com.eggloop.flow.mitl.MitlPropertiesList;
import org.junit.Test;

import static org.junit.Assert.*;

public class MitlFactoryTest {

    @Test
    public void testConstructProperties() {
        Context ns = new Context();
        new Variable("I", ns);
        MitlFactory factory = new MitlFactory(ns);
        MitlPropertiesList mitlPropertiesList = factory.constructProperties("G[3.0, 5.0] (I>0.0)\n");
        MiTL miTL = mitlPropertiesList.getProperties().get(0);
        System.out.println();
    }
}