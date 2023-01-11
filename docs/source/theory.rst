==========
The Theory
==========
The calculations performed by the functions in this package will be briefly listed here.
The reasoning behind the calculations will not be explained in detail but references to literature will be provided.

------------------------------------------------
Calculating Capacitances Using Confomal Mappings
------------------------------------------------
The general idea of conformal mappings is to transform geometries in which the field equations cannot easily be solved into ones where the solution is trivial using appropriate coordinate transformations.
Let us first broadly discuss this useful approach before moving on to the particular device geometries.

Field Volumes
=============
We consider a planar geometry of conductors that is (practically) invariant in one dimension.
By that we mean that its extent in that direction is so large that we can consider it to be infinite for the purposes of our calculations.
We will then use conformal mapping techniques to calculate the specific capacitance of the geometry (i.e. the capacitance per length in the invariant direction).
For a geometry of finite length (that is still large enough for the approximation to hold), we can the obtain its absolute capacitance by multiplying its specific capacitance with its length.

Above and beneach the conductor layer, there may be an arbitrary number of dielectric layers as well as a metal cover above the upper-most and beneath the lower-most dielectric.
Let us only consider the upper half of this system for now; the calculations for the lower half are analogous.
Let :math:`n` be the number of dielectric layers above the conductors, :math:`\varepsilon_k` the relative permittivity of the :math:`k`-th dielectric and :math:`h_k` the :math:`z`-coordinate of its upper edge::

   ───────────────────────────────────────────────────── h_n
                           eps_n
   ----------------------------------------------------- h_{n-1}
   ·                         ·                         ·
   ·                         ·                         ·
   ·                         ·                         ·
   ----------------------------------------------------- h_2
                           eps_2
   ----------------------------------------------------- h_1
                           eps_1
   ──────    ──────────       ────      ──────────────── 0

Let us call the contribution :math:`C_{\mathrm p}` of a particular region of space to the specific capacitance divided by the permittivity in that region its *field volume* :math:`v := C_{\mathrm p} / \varepsilon`.
Using conformal mappings appropriate for the conductor geometry, we can calculate the field volume :math:`v(h)` of a dielectric layer extending from the conductor layer (height :math:`0`) up to the height :math:`h`, with or without a metal cover.
The field volume of a single layer of the dielectric stack is then given by

.. math::

   v_k = v(h_k) - v(h_{k - 1})

where :math:`h_0 := 0`.
With this we can calculate the total specific capacitance of the geometry as the sum of the upper (index :math:`\mathrm u`) and lower (index :math:`\mathrm l`) contributions

.. math::

   C = \varepsilon_0 \, \Bigl( \sum_{k = 0}^{n^{\mathrm u}} v^{\mathrm u}_k \, \varepsilon^{\mathrm u}_k + \sum_{k = 0}^{n^{\mathrm l}} v^{\mathrm l}_k \, \varepsilon^{\mathrm l}_k \Bigr) \text{.}

Filling Factors and Effective Permittivity
==========================================
For further considerations (like determining the propagation velocity in a transmission line) the effective permittivity of the system can be useful.
Each dielectric contributes to it according to a filling factor:

.. math::

   \varepsilon_{\mathrm{eff}}
   := \frac{C}{C_{\mathrm{vac}}}
   = \sum_{k = 0}^{n^{\mathrm u}} f^{\mathrm u}_k \, \varepsilon^{\mathrm u}_k + \sum_{k = 0}^{n^{\mathrm l}} f^{\mathrm l}_k \, \varepsilon^{\mathrm l}_k

where :math:`C` is the total specific capacitance of the geometry and :math:`C_{\mathrm{vac}}` is its specific capacitance if all dielectrics were replaced with vacuum.
Using its vield volume, the filling factor of any of our dielectric layers is given by

.. math::

   f_k = \frac{v_k}{v_{\mathrm{tot}}}
   \qquad\text{where}\qquad
   v_{\mathrm{tot}} = v(h^{\mathrm u}_{n^{\mathrm u}}) + v(h^{\mathrm l}_{n^{\mathrm l}})
   = \sum_{k = 0}^{n^{\mathrm u}} v^{\mathrm u}_k + \sum_{k = 0}^{n^{\mathrm l}} v^{\mathrm l}_k .


--------------------------
Coplanar Waveguides (CPWs)
--------------------------
Calculating the Field Volumes
=============================
For a coplanar waveguide described by the parameters :math:`a` and :math:`b` the field volume of a dielectric layer extending from the CPW (height :math:`0`) up to the height :math:`h` is given by

.. math::

   v(h) = 2 \cdot \frac{K(m(h))}{K'(m(h))} = 2 \cdot \frac{K(m(h))}{K(1 - m(h))}

where :math:`K(m)` is the complete elliptic integral of the first kind and :math:`K'(m)` is its complement.
(Note that we are writing :math:`K` in the convention with the parameter :math:`m`, as does :func:`scipy.special.ellipk`.
Both Simons (2001) and Garg et al. (2013) use the convention with the parameter :math:`k`, where :math:`m = k^2`.)
The parameter :math:`m(h)` is given by

.. math::

   m^{\mathrm{cov}}(h) = \frac{\tanh\bigl( \frac{\pi a}{2 h} \bigr)}{\tanh\bigl( \frac{\pi b}{2 h} \bigr)}
   \qquad\text{and}\qquad
   m^{\mathrm{int}}(h) = \frac{\sinh\bigl( \frac{\pi a}{2 h} \bigr)}{\sinh\bigl( \frac{\pi b}{2 h} \bigr)}

for layers with or without a metal cover, respectively (i.e. at an interface between dielectrics or at the edge of the layer stack, respectively).
Note that

.. math::

   \lim_{h \to 0} m = 0
   \qquad\text{and}\qquad
   \lim_{h \to \infty} m = \frac ab

in both cases.

More details on these calculations can be found in Simons (2001) and Garg et al. (2013).

Further Parameters
==================
For a lossless CPW (:math:`G = 0` and :math:`R = 0` in the transmission line model) one finds wave solutions with

.. math::

   U \propto I \propto \exp(\mathrm{i} (\omega t - \gamma x))
   \quad\text{where}\quad
   \gamma = \mathrm i \omega \sqrt{L C} .

It can be shown that the phase velocity in the transmission line is given by

.. math::

   v_{\mathrm{ph}} = \frac{\omega}{\operatorname{Im}(\gamma)} = \frac{1}{\sqrt{L C}} = \frac{c}{\sqrt{\varepsilon_{\mathrm{eff}}}}

allowing us to calculate the line's characteristic impedance via

.. math::

   Z_0 = \sqrt{\frac{L}{C}} = \frac{\gamma}{\mathrm i \omega C} = \frac{1}{C v_{\mathrm{ph}}} = \frac{\sqrt{\varepsilon_{\mathrm{eff}}}}{C c}

where :math:`c` is the speed of light in vacuum.
Finally, we can calculate :math:`L` if desired:

.. math::

   L = C Z_0^2 = \frac{1}{C v_{\mathrm{ph}}^2}

------------------------------
Inderdigital Capacitors (IDCs)
------------------------------
The Approach
============
For an interdigital capacitor described by the parameters :math:`n`, :math:`\eta` and :math:`\lambda` we calculate the contributions :math:`C_{\mathrm i}` of the a between two interior electrodes and :math:`C_{\mathrm e}` of a gap next to an exterior electrode separately.
The total specific capacitance is then given by

.. math::

   C = \frac{n - 3}{2} \, C_{\mathrm i} + 2 \, \frac{C_{\mathrm i} C_{\mathrm e}}{C_{\mathrm i} + C_{\mathrm e}}

and the absolute capacitance is obtained as :math:`C_{\mathrm{abs}} = l \cdot C` with the IDC finger length :math:`l`.

While it is possible to calculate the capacitance for IDC structures with a metal cover using conformal mapping techniques similar to those used for CPW structures, Igreja (2004) does not do this and it is thus not included in this package.
The following expressions are thus only valid for dielectric layers *without* a metal cover.
In practice, this will hardly be of relevance, as the distance of a metal cover to the IDC will be much larger than the gap between the fingers (or even the IDC size), making its effects negligible.

Calculating the Field Volumes
=============================
The field volume of a dielectric layer extending from the CPW (height :math:`0`) up to the height :math:`h` corresponding to one of those gaps is given by

.. math::

   v(h) = \frac{K(m(h))}{K'(m(h))} = \frac{K(m(h))}{K(1 - m(h))}

where :math:`K(m)` is the complete elliptic integral of the first kind and :math:`K'(m)` is its complement.
(Note that we are writing :math:`K` in the convention with the parameter :math:`m`, as does :func:`scipy.special.ellipk`.
Igreja (2004) uses the convention with the parameter :math:`k`, where :math:`m = k^2`.)

For interior electrodes :math:`m(h)` is given by

.. math::

   m = t_2^2 \, \frac{t_4^2 - 1}{t_4^2 - t_2^2}
   \qquad,\qquad
   \lim_{h \to \infty} m = \sin\Bigl( \frac{\pi \eta}{2} \Bigr)^2 ,

where

.. math::

   t_2 = \operatorname{sn}(K(k^2) \, \eta, k^2)
   \quad,\quad
   t_4 = \frac{1}{k}
   \quad,\quad
   k = \Bigl( \frac{\vartheta_2(0, q)}{\vartheta_3(0, q)} \Bigr)^2
   \quad,\quad
   q = \exp\Bigl(- \frac{4 \pi h}{\lambda}\Bigr)

with the elliptic sine (Jacobi elliptic function *sinus amplitudinis*) :math:`\operatorname{sn}` and the Jacobi theta functions :math:`\vartheta_i`.
As for :math:`K` we use the :math:`m`-convention for :math:`\operatorname{sn}`, as does :func:`scipy.special.ellipj`, whereas Igreja (2004) uses the :math:`k`-convention.

For exterior electrodes one finds

.. math::

   m = \frac{1}{t_3^2} \frac{t_4^2 - t_3^2}{t_4^2 - 1}
   \qquad,\qquad
   \lim_{h \to \infty} m = \frac{4 \eta}{(1 + \eta)^2} ,

where

.. math::

   t_3 = \cosh\Bigl( \frac{\pi (1 - \eta) \lambda}{8 h} \Bigr)
   \quad,\quad
   t_4 = \cosh\Bigl( \frac{\pi (1 + \eta) \lambda}{8 h} \Bigr) .

More details on these calculations can be found in Igreja (2004).

----------
References
----------
- Garg, R., Bahl, I., & Bozzi, M. (2013) *Microstrip Lines and Slotlines* (3rd ed.). Artech House.
- Igreja, R., Dias, C. J. (2004) Analytical evaluation of the interdigital electrodes capacitance for a multi-layered structure. *Sensors and Actuators A: Physical 112*, 291--301.
- Simons, R. N. (2001). *Coplanar Waveguide Circuits, Components, and Systems.* John Wiley & Sons.
