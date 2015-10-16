
import os
import time
import numpy as np
import logging
from string import digits

from PGL.components.airfoil import AirfoilShape


class CS2DtoBECAS(object):
    """
    Component that generates a set of BECAS input files based on
    a fusedwind.turbine.structure_vt.CrossSectionStructureVT.

    This class uses shellexpander, which comes as part of the BECAS distribution.

    parameters
    ----------
    config: dict
        dictionary of model specific inputs
    cs2d: dict
        dictionary containing coordinates and materials
    path_shellexpander: str
        Absolute path to shellexpander.py
    total_points: int
        Number of total geometry points to define the shape of the cross-section
    max_layers: int
        Maximum number of layers through thickness
    open_te: bool
        If True, TE will be left open
    becas_inputs: str
        Relative path for the future BECAS input files
    section_name: str
        Section name used by shellexpander, also by BECASWrapper
    dominant_elsets: list
        list of region names defining the spar cap regions for correct meshing
    path_input: str
        path to the generated BECAS input files
    airfoil: array
        the re-distributed airfoil coordinates
    web_coord: array
        the distributed shear web coordinates
    nodes: array
        all the nodes, web+airfoil for the 2D cross section
    te_ratio: float
        ratio between outer TE planform and TE layup thickness
    thickness_ratio: array
        ratio between outer surface height and layup thickness at DPs
    """

    def __init__(self, cs2d, **kwargs):

        self.path_shellexpander = ''

        self.dry_run = False
        self.cs2d = cs2d
        self.total_points = 100
        self.max_layers = 0
        self.open_te = False
        self.becas_inputs = 'becas_inputs'
        self.section_name = 'BECAS_SECTION%3.3f' % cs2d['s']
        self.dom_regions = ['REGION04', 'REGION08']

        self.path_input = ''
        self.airfoil = np.array([])
        self.web_coord = np.array([])
        self.nodes = np.array([])
        self.elset_defs = {}
        self.elements = np.array([])
        self.nodes_3d = np.array([])
        self.el_3d = np.array([])
        self.te_ratio = 0.
        self.thickness_ratio = np.array([])


        for k, w in kwargs.iteritems():
            try:
                setattr(self, k, w)
            except:
                pass

    def compute(self, redistribute_flag=True):
        """  """

        tt = time.time()

        self.redistribute_flag = redistribute_flag

        if self.path_shellexpander == '':
            raise RuntimeError('path_shellexpander not specified')

        self.path_input = os.path.join(self.becas_inputs, self.section_name)

        if self.redistribute_flag:
            self.iDPs = []
        self.DPs01 = []
        self.web_coord = np.array([])
        self.web_to_airfoil_idx = []  # node index on airfoil surface where webs are attached
                                      # corresponding to cs2d['web_def']
        self.web_element_idx = []   # web element indices
        self.webDPs = []

        self.coords = AirfoilShape(points=self.cs2d['coords'])

        self.compute_max_layers()
        self.compute_airfoil()
            # self.output_te_ratio()
            # self.compute_thickness_to_airfoil_ratio()
        self.add_shearweb_nodes()
        self.create_elements()
        self.create_elements_3d(reverse_normals=False)
        self.write_abaqus_inp()
        self.write_becas_inp()


    def compute_max_layers(self):
        """
        The number of elements used to discretize the shell thickness.
        The minimum value is the largest number of layers of different
        material anywhere in the airfoil.
        """

        self.max_thickness = 0.
        for r in self.cs2d['regions']:
            self.max_layers = max(self.max_layers, len(r['layers']))
            self.max_thickness = np.maximum(sum(r['thicknesses']), self.max_thickness)


    def compute_airfoil(self):
        """Redistribute mesh points evenly among regions

        After defining different regions this method will assure that, given
        a total number of mesh points, the cell size in each region is similar.
        The region boundaries and shear web positions will be approximately
        on the same positions as defined.

        Region numbers can be added arbitrarely.
        """

        af = self.coords

        if np.linalg.norm(af.points[0] - af.points[-1]) > 0.:
            self.open_te = True

        # construct distfunc for airfoil curve
        dist = []

        # compute cell size
        # and adjust max region thickness
        ds_const = af.smax / self.total_points
        if ds_const < 1.2 * self.max_thickness:
            ds_old = ds_const
            new_ds = 1.2 * self.max_thickness
            self.total_points = np.maximum(int(af.smax / new_ds), 70)
            ds_const = af.smax / self.total_points
            # self._logger.info('increasing cell size from %5.3f to %5.3f '
            #     'and reducing number of elements to %i' % (ds_const, ds_old, self.total_points))
            print('increasing cell size from %5.3f to %5.3f '
                'and reducing number of elements to %i' % (ds_const, ds_old, self.total_points))
        self.ds_const = ds_const / af.smax

        # convert DPs to s01 notation
        self.DPs01 = [af.s_to_01(s) for s in self.cs2d['DPs']]
        self.DPcoords = [af.interp_s(s) for s in self.DPs01]

        # full redistribution of nodes
        if self.redistribute_flag:
            dist_ni = 0
            s_start = 0.
            self.iDPs = []
            for i, s in enumerate(self.DPs01):
                s_end = s
                dist_ni += max(1, int(round( (s_end - s_start)/self.ds_const )))
                # add a distribution point to the Curve
                dist.append([s_end, self.ds_const, dist_ni])
                self.iDPs.append(dist_ni-1)
                s_start = s

            self.dist = dist
            self.dist_ni = dist_ni

        # maintain nodal distribution only changing
        # the DP position `s`.
        else:
            for i, s in enumerate(self.DPs01):
                self.dist[i][0] = s

        afn = af.redistribute(self.dist_ni, dist=dist)
        self.airfoil = afn.points

        self.total_points = self.airfoil.shape[0]
        print 'total_points final', self.total_points

    def add_shearweb_nodes(self):
        """
        Distribute nodes over the shear web. Use the same spacing as used for
        the airfoil nodes.
        """


        # find the thickness in the TE region to close with a web from first and last region
        r_TE_pres = self.cs2d['regions'][0]
        r_TE_suc = self.cs2d['regions'][-1]
        tpres = sum(r_TE_pres['thicknesses'])
        tsuc = sum(r_TE_pres['thicknesses'])
        # add a 50% thickness safety factor
        TE_thick_max = (tpres + tsuc) * 1.5
        self.web_coord = np.array([])
        element_counter = self.total_points
        for i, w in enumerate(self.cs2d['webs']):
            if sum(w['thicknesses']) == 0.:
                self.web_element_idx.append([])
                self.web_to_airfoil_idx.append([])
                i0 = self.cs2d['web_def'][i][0]
                i1 = self.cs2d['web_def'][i][1]
                self.webDPs.append([self.cs2d['DPs'][i0], self.cs2d['DPs'][i1]])
                continue
            ds_mean = np.maximum(self.ds_const * self.coords.smax, self.max_thickness * 1.2)
            i0 = self.cs2d['web_def'][i][0]
            i1 = self.cs2d['web_def'][i][1]
            self.webDPs.append([self.cs2d['DPs'][i0], self.cs2d['DPs'][i1]])
            node1 = self.DPcoords[i0]
            node2 = self.DPcoords[i1]
            self.web_to_airfoil_idx.append([self.iDPs[i0], self.iDPs[i1]])
            # the length of the shear web is then
            len_web = np.linalg.norm( node1-node2 )
            nr_points = max(int(round(len_web / ds_mean, 0)), 3)
            # generate nodal coordinates on the shear web
            if TE_thick_max > len_web:
                # if a web is used to close the TE and the the web is very
                # short, no extra nodes are placed along the web to avoid mesh issues
                x = np.array([node1[0], node2[0]])
                y = np.array([node1[1], node2[1]])
                self.web_element_idx.append([])
            else:
                x = np.linspace(node1[0], node2[0], nr_points)
                y = np.linspace(node1[1], node2[1], nr_points)
                nr_points = x.shape[0]
                # self.nr_web_nodes.append(nr_points)
                # and add them to the shear web node collection, but ignore the
                # first and last nodes because they are already included in
                # the airfoil coordinates.
                tmp = np.ndarray((len(x)-2, 2))
                tmp[:,0] = x[1:-1]
                tmp[:,1] = y[1:-1]
                i0 = element_counter
                i1 = (nr_points - 2) + i0 - 1
                self.web_element_idx.append([i0, i1])
                element_counter += nr_points - 2
                # remember to start and stop indices for the shear web nodes
                try:
                    self.web_coord = np.append(self.web_coord, tmp, axis=0)
                except:
                    self.web_coord = tmp.copy()

    def create_elements(self, debug=False):
        """
        Create the elements and assign element sets to the different regions.

        Assign node and element numbers for the current airfoil points and
        shear webs. Since the airfoil coordinates are ordered clockwise
        continuous the node and element numbering is trivial.

        Note when referring to node and element numbers array indices are used.
        BECAS uses 1-based counting instead of zero-based.
        """

        # by default, the node and element numbers are zero based numbering
        self.onebasednumbering = False

        # element numbers for each ELSET
        self.elset_defs = {}

        nr_air_n = len(self.airfoil)
        nr_web_n = len(self.web_coord)
        nr_nodes = nr_air_n + nr_web_n
        # for closed TE, nr_elements = nr_nodes, for open TE, 1 element less
        nr_web = 0
        for i in range(len(self.cs2d['webs'])):
            if sum(self.cs2d['webs'][i]['thicknesses']) > 0.:
                nr_web += 1
        if self.open_te:
            nr_elements = nr_nodes + nr_web - 1
            nr_air_el = nr_air_n - 1
        else:
            nr_elements = nr_nodes + nr_web
            nr_air_el = nr_air_n

        # place all nodal coordinates in one array. The elements are defined
        # by the node index.
        self.nodes = np.zeros( (nr_nodes, 3) )
        self.nodes[:nr_air_n,:2] = self.airfoil[:,:]
        self.nodes[nr_air_n:,:2] = self.web_coord

        # Elements are bounded by two neighbouring nodes. By closing the
        # circle (connecting the TE suction side with the pressure side), we
        # have as many elements as there are nodes on the airfoil
        # elements[element_nr, (node1,node2)]: shape=(n,2)
        # for each web, we have nr_web_nodes+1 number of elements
        self.elements = np.ndarray((nr_elements, 2), dtype=np.int)
        if self.open_te:
            self.elements[:nr_air_el,0] = np.arange(nr_air_n-1, dtype=np.int)
            self.elements[:nr_air_el,1] = self.elements[:nr_air_el, 0] + 1
        else:
            # when the airfoil is closed, add one node number too much...
            self.elements[:nr_air_el,0] = np.arange(nr_air_n, dtype=np.int)
            self.elements[:nr_air_el,1] = self.elements[:nr_air_el,0] + 1
            # last node on last element is first node, airfoil is now closed
            self.elements[nr_air_el-1,1] = 0

        if debug:
            print 'nr airfoil nodes: %4i' % (len(self.airfoil))
            print '    nr web nodes: %4i' % len(self.web_coord)

        web_el = []
        pre_side, suc_side = [], []

        # compute TE panel angle
        v0 = np.array(self.airfoil[1] - self.airfoil[0])
        v1 = np.array(self.airfoil[-2] - self.airfoil[-1])
        self.TEangle = np.arccos(np.dot(v0,v1) / (np.linalg.norm(v0)*np.linalg.norm(v1)))*180./np.pi

        # self._logger.info('TE angle = %3.3f' % self.TEangle)

        el_offset = nr_air_el
        # define el for each shear web, and create corresponding node groups
        for i, w in enumerate(self.cs2d['webs']):
            w_name = 'WEB%02d' % i
            if sum(w['thicknesses']) == 0.: continue

            # starting index of web elements
            iw_start = el_offset

            # number of intermediate shear web nodes
            try:
                interior_nodes = self.web_element_idx[i][1] - self.web_element_idx[i][0] + 1
            except:
                interior_nodes = 0

            # end index of web elements
            iw_end = interior_nodes + iw_start

            if interior_nodes > 0:
                # first element: airfoil surface to first interior web node
                self.elements[iw_start,:] = [self.web_to_airfoil_idx[i][0], self.web_element_idx[i][0]]

                # elements in between
                wnodes = np.arange(self.web_element_idx[i][0], self.web_element_idx[i][1], dtype=np.int)
                self.elements[iw_start+1:iw_end, 0] = wnodes
                self.elements[iw_start+1:iw_end, 1] = wnodes + 1

                # final element that connects the web back to the airfoil
                self.elements[iw_end, :] = [self.web_element_idx[i][-1], self.web_to_airfoil_idx[i][1]]
            else:
                # web contains only a single element
                print 'single web element'
                self.elements[iw_start,:] = [self.web_to_airfoil_idx[i][0], self.web_to_airfoil_idx[i][1]]


            # and now we can populate the different regions with their
            # corresponding elements
            if self.webDPs[i][0] in [-1., 1.] and abs(self.TEangle) > 150.:
                self.elset_defs[w_name] = np.arange(el_offset, el_offset+interior_nodes + 1, dtype=np.int)
                suc_side.extend(range(el_offset, el_offset+interior_nodes + 1))
                print('TEangle > 150, adding to suc_side! s=%3.3f %s' % (self.cs2d['s'], w_name))
            else:
                self.elset_defs[w_name] = np.arange(el_offset, el_offset+interior_nodes + 1, dtype=np.int)
                web_el.extend(range(el_offset, el_offset+interior_nodes + 1))
            el_offset += interior_nodes + 1

        if len(web_el) > 0:
            self.elset_defs['WEBS'] = np.array(web_el, dtype=np.int)

        # element groups for the regions
        for i, r in enumerate(self.cs2d['regions']):
            r_name = 'REGION%02d' % i
            # do not include element r.s1_i, that is included in the next elset
            self.elset_defs[r_name] = np.arange(self.iDPs[i], self.iDPs[i+1], dtype=np.int)

            # group in suction and pressure side (s_LE=0)
            if self.cs2d['DPs'][i + 1] <= 0:
                pre_side.extend(range(self.iDPs[i], self.iDPs[i+1]))
            else:
                suc_side.extend(range(self.iDPs[i], self.iDPs[i+1]))

        tmp = np.array(list(pre_side)+list(suc_side))
        pre0, pre1 = tmp.min(), tmp.max()
        self.elset_defs['SURFACE'] = np.arange(pre0, pre1+1, dtype=np.int)

    def create_elements_3d(self, reverse_normals=False):
        """
        Shellexpander wants a 3D section as input. Create a 3D section
        which is just like the 2D version except with a depth defined as 1%
        of the chord length.
        """

        # Compute depth of 3D mesh as 1% of chord lenght
        depth = -0.01 * self.coords.chord
        if reverse_normals:
            depth = depth * (-1.0)

        nr_nodes_2d = len(self.nodes)
        # Add nodes for 3D mesh
        self.nodes_3d = np.ndarray( (nr_nodes_2d*2, 3) )
        self.nodes_3d[:nr_nodes_2d, :] = self.nodes
        self.nodes_3d[nr_nodes_2d:, :] = self.nodes + np.array([0,0,depth])
        # Generate shell elements
        self.el_3d = np.ndarray( (len(self.elements), 4), dtype=np.int)
        self.el_3d[:,:2] = self.elements
        self.el_3d[:,2] = self.elements[:,1] + nr_nodes_2d
        self.el_3d[:,3] = self.elements[:,0] + nr_nodes_2d

    def one_based_numbering(self):
        """
        instead of 0, 1 is the first element and node number. All nodes and
        elements +1.

        Note that this does not affect the indices defined in the region
        and web attributes
        """
        if not self.onebasednumbering:

            self.elements += 1
            self.el_3d += 1
            for elset in self.elset_defs:
                self.elset_defs[elset] += 1

            self.onebasednumbering = True

    def zero_based_numbering(self):
        """
        switch back to 0 as first element and node number
        """

        if self.onebasednumbering:

            self.elements -= 1
            self.el_3d -= 1
            for elset in self.elset_defs:
                self.elset_defs[elset] -= 1

            self.onebasednumbering = False

    def write_abaqus_inp(self, fname=False):
        """Create Abaqus inp file which will be served to shellexpander so
        the actual BECAS input can be created.
        """

        def write_n_int_per_line(list_of_int, f, n):
            """Write the integers in list_of_int to the output file - n integers
            per line, separated by commas"""
            i=0
            for number in list_of_int:
                i=i+1
                f.write('%d' %(number ))
                if i < len(list_of_int):
                    f.write(',  ')
                if i%n == 0:
                    f.write('\n')
            if i%n != 0:
                f.write('\n')

        self.abaqus_inp_fname = 'airfoil_abaqus.inp'

        # FIXME: for now, force 1 based numbering, I don't think shellexpander
        # and/or BECAS like zero based node and element numbering
        self.one_based_numbering()

        # where to start node/element numbering, 0 or 1?
        if self.onebasednumbering:
            off = 1
        else:
            off = 0

        with open(self.abaqus_inp_fname, 'w') as f:

            # Write nodal coordinates
            f.write('**\n')
            f.write('********************\n')
            f.write('** NODAL COORDINATES\n')
            f.write('********************\n')
            f.write('*NODE\n')
            tmp = np.ndarray( (len(self.nodes_3d),4) )
            tmp[:,0] = np.arange(len(self.nodes_3d), dtype=np.int) + off
            tmp[:,1:] = self.nodes_3d
            np.savetxt(f, tmp, fmt='%1.0f, %1.20e, %1.20e, %1.20e')

            # Write element definitions
            f.write('**\n')
            f.write('***********\n')
            f.write('** ELEMENTS\n')
            f.write('***********\n')
            f.write('*ELEMENT, TYPE=S4, ELSET=%s\n' % self.section_name)
            tmp = np.ndarray( (len(self.el_3d),5) )
            tmp[:,0] = np.arange(len(self.el_3d), dtype=np.int) + off
            tmp[:,1:] = self.el_3d
            np.savetxt(f, tmp, fmt='%i, %i, %i, %i, %i')

            # Write new element sets
            f.write('**\n')
            f.write('***************\n')
            f.write('** ELEMENT SETS\n')
            f.write('***************\n')
            for elset in sorted(self.elset_defs.keys()):
                elements = self.elset_defs[elset]
                f.write('*ELSET, ELSET=%s\n' % (elset))
#                np.savetxt(f, elements, fmt='%i', delimiter=', ')
                write_n_int_per_line(list(elements), f, 8)

            # Write Shell Section definitions
            # The first layer is the outer most layer.
            # The second item ("int. points") and the fifth item ("plyname")
            # are not relevant. The are kept for compatibility with the ABAQUS
            # input syntax. As an example, take this one:
            # [0.006, 3, 'TRIAX', 0.0, 'Ply01']
            f.write('**\n')
            f.write('****************************\n')
            f.write('** SHELL SECTION DEFINITIONS\n')
            f.write('****************************\n')
            names = ['REGION%02d' % i for i in range(len(self.cs2d['regions']))]
            names.extend(['WEB%02d' % i for i in range(len(self.cs2d['webs']))])
            for i, r in enumerate(self.cs2d['regions'] + self.cs2d['webs']):
                r_name = names[i]
                if r_name.startswith('WEB'):
                    offset = 0.0
                else:
                    offset = -0.5
                text = '*SHELL SECTION, ELSET=%s, COMPOSITE, OFFSET=%3.3f\n'
                f.write(text % (r_name, offset))
                for il, l_name in enumerate(r['layers']):

                    materialname = l_name.translate(None, digits).lower()
                    m_ix = self.cs2d['materials'][materialname]
                    if self.cs2d['failcrit'][m_ix] == 'maximum_stress':
                        mname = materialname + 'MAXSTRESS'
                    elif self.cs2d['failcrit'][m_ix] == 'maximum_strain':
                        mname = materialname + 'MAXSTRAIN'
                    elif self.cs2d['failcrit'][m_ix] == 'tsai_wu':
                        mname = materialname + 'TSAIWU'
                    else:
                        mname = materialname
                    plyname = 'ply%02d' % i

                    layer_def = (r['thicknesses'][il], 3, mname,
                                 r['angles'][il], plyname)
                    f.write('%g, %d, %s, %g, %s\n' % layer_def )

            # Write material properties
            f.write('**\n')
            f.write('**********************\n')
            f.write('** MATERIAL PROPERTIES\n')
            f.write('**********************\n')
            for matname, ix in self.cs2d['materials'].iteritems():
                md = self.cs2d['matprops'][ix]

                if self.cs2d['failcrit'][ix] == 'maximum_stress':
                    mname = matname + 'MAXSTRESS'
                elif self.cs2d['failcrit'][ix] == 'maximum_strain':
                    mname = matname + 'MAXSTRAIN'
                elif self.cs2d['failcrit'][ix] == 'tsai_wu':
                    mname = matname + 'TSAIWU'
                else:
                    mname = matname
                f.write('*MATERIAL, NAME=%s\n' % (mname))
                f.write('*ELASTIC, TYPE=ENGINEERING CONSTANTS\n')
                f.write('%g, %g, %g, %g, %g, %g, %g, %g\n' % (md[0], md[1],
                    md[2], md[3], md[4], md[5], md[6], md[7]))
                f.write('%g\n' % (md[8]))
                f.write('*DENSITY\n')
                f.write('%g\n' % (md[9]))
                # failcrit array
                # s11_t s22_t s33_t s11_c s22_c s33_c
                # t12 t13 t23 e11_t e22_t e33_t e11_c e22_c e33_c g12 g13 g23
                # gM0 C1a C2a C3a C4a
                md = self.cs2d['failmat'][ix]
                f.write('*FAIL STRESS\n')
                # gMa = gM0 C1a C2a C3a C4a
                gMa = md[18] * (md[19] + md[20] + md[21] + md[22])
                f.write('%g, %g, %g, %g, %g\n' % (gMa * md[0], gMa * md[3],
                                                  gMa * md[1], gMa * md[4], gMa * md[6]))
                f.write('*FAIL STRAIN\n')
                f.write('%g, %g, %g, %g, %g\n' % (gMa * md[9], gMa * md[12],
                                                  gMa * md[10], gMa * md[13], gMa * md[15]))
                f.write('**\n')
        print 'Abaqus input file written: %s' % self.abaqus_inp_fname

    def write_becas_inp(self):
        """
        When write_abaqus_inp has been executed we have the shellexpander
        script that can create the becas input

        Dominant regions should be the spar caps.
        """

        class args: pass
        # the he name of the input file containing the finite element shell model
        args.inputfile = self.abaqus_inp_fname #--in
        # The element sets to be considered (required). If more than one
        # element set is given, nodes appearing in more the one element sets
        # are considered "corners". Should be pressureside, suction side and
        # the webs
        elsets = []
        target_sets = ['SURFACE', 'WEBS']
        for elset in target_sets:
            if elset in self.elset_defs:
                elsets.append(elset)
        if len(elsets) < 1:
            raise ValueError, 'badly defined element sets'
        args.elsets = elsets #--elsets, list
        args.sections = self.section_name #--sec
        args.layers = self.max_layers #--layers
        args.nodal_thickness = 'min' #--ntick, choices=['min','max','average']
        args.dominant_elsets = self.dominant_elsets #--dom, list
        args.centerline = None #--cline, string
        args.becasdir = self.becas_inputs #--bdir
        args.debug = False #--debug, if present switch to True

        if not self.dry_run:
            import imp
            shellexpander = imp.load_source('shellexpander',
                              os.path.join(self.path_shellexpander, 'shellexpander.py'))

            shellexpander.main(args)

    def output_te_ratio(self):
        """
        outputs a ratio between the thickness of the trailing edge panels
        and the thickness of the trailing edge

        """

        if not self.open_te:
            self.te_ratio = 0.
            return

        # pressure and suction side panels
        dTE = np.abs(self.airfoil[-1, 1] - self.airfoil[0, 1]) / 2.
        r_name = self.cs2d.regions[-1]
        r_TE_suc = getattr(self.cs2d, r_name)
        r_name = self.cs2d.regions[0]
        r_TE_pres = getattr(self.cs2d, r_name)
        thick_max = (r_TE_pres.thickness + r_TE_suc.thickness) / 2.
        self.te_ratio = thick_max / dTE
        self._logger.info('TE ratio %f %f %f' % (self.cs2d.s, dTE * 3., self.te_ratio))

    def compute_thickness_to_airfoil_ratio(self):
        """
        compute the ratio between the material thickness and the airfoil
        thickness at the different DPs
        """
        # extract number of dominant region from name REGION##
        nr_cap = int(self.dominant_elsets[0][6:])
        self.thickness_ratio = np.zeros(nr_cap+1)
        # loop DP points up to the cap from the trailing edge
        for i in range(nr_cap+1):
            # pick DP points that now instead are called CPs
            x1, y1 = self.DPs01[i]
            x2, y2 = self.DPs01[-(i+1)]
            # compute shape thickness
            shape_thickness = ((x1 - x2)**2 + (y1 - y2)**2)**.5
            r_name1 = self.cs2d.regions[i]
            r_1_suc = getattr(self.cs2d, r_name1)
            r_name2 = self.cs2d.regions[-(i+1)]
            r_2_pres = getattr(self.cs2d, r_name2)
            thick_max = (r_1_suc.thickness + r_2_pres.thickness)
            self.thickness_ratio[i] = thick_max / shape_thickness
